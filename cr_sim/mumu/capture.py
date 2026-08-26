"""Screen capture, and recording a session of it to disk for offline
comparison against the simulator.

**Why ``exec-out`` and not ``adb shell screencap -p`` redirected to a file.**
``adb shell`` allocates a pty-like channel, and on Windows that channel
translates every ``0x0A`` byte to ``0x0D 0x0A`` on the way through -- fine for
text, silent corruption for a binary PNG, where roughly 1 byte in 256 is
``0x0A`` by chance. ``adb exec-out`` runs the same remote command without that
channel, so the PNG bytes arrive byte-for-byte intact. That guarantee only
holds if the pipe is read as raw bytes the whole way, which is why
:meth:`cr_sim.mumu.adb.AdbBridge.exec_out` always invokes its runner with
``binary=True`` -- decoding as text anywhere in the path would reintroduce the
same corruption ``exec-out`` exists to avoid. This module never uses
``adb shell screencap`` for that reason; there is no CRLF-stripping fallback
path here to keep untested.

**Imaging library.** No hard dependency is added for PNG decoding -- whatever
is already importable is used, in order: Pillow, then OpenCV (``cv2``), then
``imageio``. On the machine this module was written on, Pillow 12.2.0 and
OpenCV 4.13.0 are both installed; ``imageio`` is not. If none of the three are
importable, decoding fails with :class:`MissingImagingLibraryError` rather
than an opaque ``ImportError`` from three levels down.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adb import AdbBridge

__all__ = [
    "MissingImagingLibraryError",
    "screencap",
    "RecordingSummary",
    "FrameRecorder",
]


class MissingImagingLibraryError(RuntimeError):
    """No PNG-capable imaging library is importable in this environment."""


def _decode_png(data: bytes) -> np.ndarray:
    """PNG bytes -> an (H, W, 3) uint8 RGB array."""
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    try:
        import cv2
    except ImportError:
        pass
    else:
        encoded = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)  # BGR, drops alpha
        if decoded is None:
            raise MissingImagingLibraryError("cv2.imdecode failed to parse the screencap PNG")
        return np.ascontiguousarray(decoded[:, :, ::-1])  # BGR -> RGB

    try:
        import imageio.v3 as iio
    except ImportError:
        pass
    else:
        frame = np.asarray(iio.imread(data))
        return np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)  # drop alpha if present

    raise MissingImagingLibraryError(
        "none of Pillow, OpenCV (cv2) or imageio is importable in this environment; "
        "install one of them to decode screencap PNGs"
    )


def _save_frame(path: Path, frame: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        Image.fromarray(frame).save(path)
        return

    try:
        import cv2
    except ImportError:
        pass
    else:
        cv2.imwrite(str(path), frame[:, :, ::-1])  # RGB -> BGR for cv2's writer
        return

    raise MissingImagingLibraryError(
        "neither Pillow nor OpenCV (cv2) is importable in this environment; "
        "install one of them to write captured frames to disk"
    )


def screencap(bridge: AdbBridge) -> np.ndarray:
    """One frame from the emulator, as an (H, W, 3) uint8 RGB array."""
    data = bridge.exec_out("screencap -p")
    return _decode_png(data)


@dataclass(slots=True)
class RecordingSummary:
    """What actually happened during a :meth:`FrameRecorder.record` run, as
    typed data -- the same numbers :meth:`FrameRecorder.record` also writes to
    the session's JSON sidecar."""

    frame_count: int
    elapsed_s: float
    achieved_fps: float
    output_dir: Path


class FrameRecorder:
    """Captures frames to a directory at a best-effort target rate.

    ``screencap`` over adb is slow -- commonly reported in the 2-8 fps range
    even on capable hardware, because each frame is a full on-device PNG
    encode plus a round trip over USB or TCP, not a video stream. Whether
    frame-by-frame comparison against this simulator's 60-tick-per-second
    engine is achievable at all depends on the real number for a given
    machine and emulator, which is why :meth:`record` always reports
    ``achieved_fps`` measured from wall-clock elapsed time rather than
    assuming ``target_fps`` was met. This module was written on a machine
    with no MuMu install, so no real figure could be measured here; treat any
    number that has not been produced by an actual run on a real device the
    same way ``reference/anchors.json`` treats an unconfirmed figure.
    """

    def __init__(
        self,
        bridge: AdbBridge,
        output_dir: str | Path,
        *,
        target_fps: float = 5.0,
        image_format: str = "png",
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.bridge = bridge
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.image_format = image_format

    def record(
        self,
        *,
        duration_s: float | None = None,
        max_frames: int | None = None,
    ) -> RecordingSummary:
        """Capture until ``duration_s`` elapses or ``max_frames`` frames have
        been written, whichever comes first. At least one of the two must be
        given, or the loop would never stop.

        Filenames carry a monotonic offset from the start of the recording,
        not a wall-clock timestamp: wall-clock time can jump mid-recording
        (NTP sync, a DST transition) in a way that would break the ordering
        the filenames exist to preserve, and an epoch-millisecond name would
        be needlessly long for a session that in practice never runs more
        than a few minutes at this frame rate.
        """
        if duration_s is None and max_frames is None:
            raise ValueError("record() needs duration_s or max_frames, or it never stops")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        start = time.monotonic()
        frame_count = 0
        first_frame: np.ndarray | None = None
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break
            if duration_s is not None and (time.monotonic() - start) >= duration_s:
                break

            frame = screencap(self.bridge)
            if first_frame is None:
                first_frame = frame
            elapsed_ms = int((time.monotonic() - start) * 1000)
            filename = self.output_dir / f"frame_{elapsed_ms:010d}.{self.image_format}"
            _save_frame(filename, frame)
            frame_count += 1

            # Best-effort pacing toward target_fps. screencap's own latency
            # usually exceeds the interval anyway (see the class docstring),
            # so this is frequently a no-op; achieved_fps below, not this
            # target, is the number that actually describes what happened.
            remaining = start + frame_count / self.target_fps - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        elapsed = time.monotonic() - start
        achieved_fps = frame_count / elapsed if elapsed > 0 else 0.0

        sidecar = {
            "target_fps": self.target_fps,
            "achieved_fps": achieved_fps,
            "frame_count": frame_count,
            "elapsed_s": elapsed,
            "image_format": self.image_format,
            "device_resolution": list(first_frame.shape[1::-1]) if first_frame is not None else None,
            "bridge": {"host": self.bridge.host, "port": self.bridge.port},
        }
        (self.output_dir / "session.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )

        return RecordingSummary(
            frame_count=frame_count,
            elapsed_s=elapsed,
            achieved_fps=achieved_fps,
            output_dir=self.output_dir,
        )
