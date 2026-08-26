"""A thin wrapper over the ``adb`` command-line client, targeting MuMu.

Every other module in this package goes through :class:`AdbBridge` rather
than shelling out directly, for one reason: it is the only place that needs
to know where the ``adb`` binary actually lives, and that turns out to vary a
lot machine to machine.

**Locating adb.** PATH is checked first -- a system-wide adb (Android
Studio's platform-tools, or one installed by hand) is more likely to be
current than whatever an emulator installer bundled years ago. The fallback
list covers the install layouts MuMu has actually shipped:

* MuMu Player 12, current NetEase branding:
  ``C:\\Program Files\\Netease\\MuMu Player 12\\shell\\adb.exe``
* An earlier MuMu Player 12 installer used this directory name instead:
  ``C:\\Program Files\\Netease\\MuMuPlayer-12.0\\shell\\adb.exe``
* MuMu 6, the pre-"Player 12" line, shipped its own adb-compatible binary
  under a different name rather than a plain ``adb.exe``, but it accepts the
  same command line:
  ``C:\\Program Files (x86)\\NetEase\\MuMu\\emulator\\nemu\\vmonitor\\bin\\adb_server.exe``

Checked directly on the machine this module was written on: none of the
three paths exist, ``adb`` is not on PATH, and there is no ``Netease`` or
``NetEase`` directory under either Program Files -- MuMu is not installed
here. :func:`find_adb_binary` and every :class:`AdbBridge` method that needs
a binary are written and tested against an injected fake runner precisely
because of that; nothing in this module's test suite requires a real
emulator.

**Ports.** MuMu's single-instance default is 7555. MuMu Player 12's
multi-instance manager numbers its instances 16384, 16416, 16448, ... (+32
per instance) -- see :func:`instance_port`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Sequence

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "AdbDevice",
    "AdbError",
    "AdbNotFoundError",
    "AdbTimeoutError",
    "AdbCommandError",
    "find_adb_binary",
    "instance_port",
    "AdbBridge",
]

#: MuMu's default single-instance ADB endpoint.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7555

#: See the module docstring for what each of these corresponds to. Order
#: matters only in that PATH (checked separately, first) always wins.
_CANDIDATE_ADB_PATHS: tuple[str, ...] = (
    r"C:\Program Files\Netease\MuMu Player 12\shell\adb.exe",
    r"C:\Program Files\Netease\MuMuPlayer-12.0\shell\adb.exe",
    r"C:\Program Files (x86)\NetEase\MuMu\emulator\nemu\vmonitor\bin\adb_server.exe",
)

#: MuMu Player 12 multi-instance base port and per-instance stride, per its
#: own instance manager UI.
_MULTI_INSTANCE_BASE_PORT = 16384
_MULTI_INSTANCE_PORT_STRIDE = 32


def instance_port(index: int) -> int:
    """The ADB port MuMu Player 12's multi-instance manager assigns to
    instance ``index`` (0-based): 16384, 16416, 16448, ..."""
    if index < 0:
        raise ValueError("instance index must be >= 0")
    return _MULTI_INSTANCE_BASE_PORT + index * _MULTI_INSTANCE_PORT_STRIDE


def find_adb_binary(extra_candidates: Iterable[str | Path] = ()) -> Path | None:
    """Locate an adb-compatible binary, or ``None`` if there is not one.

    ``extra_candidates`` lets a caller (or a test) add paths ahead of a fresh
    search without needing to touch the module-level fallback list.
    """
    found = shutil.which("adb")
    if found:
        return Path(found)
    for candidate in (*extra_candidates, *_CANDIDATE_ADB_PATHS):
        path = Path(candidate)
        if path.is_file():
            return path
    return None


class AdbDevice(NamedTuple):
    """One line of ``adb devices`` output."""

    serial: str
    #: "device" (ready), "offline", "unauthorized", etc.
    state: str


class AdbError(RuntimeError):
    """Base class for every failure this module raises deliberately."""


class AdbNotFoundError(AdbError):
    """No adb binary on PATH and none at any known MuMu install location."""


class AdbTimeoutError(AdbError):
    """adb did not return within the configured timeout.

    A hung adb process (a dead server, a MuMu window that stopped responding)
    should fail loudly and promptly rather than block a capture loop
    indefinitely, which is why every subprocess call in this module passes an
    explicit timeout.
    """


class AdbCommandError(AdbError):
    """adb ran and reported failure. Carries stderr for diagnosis."""

    def __init__(self, command: Sequence[str], returncode: int, stderr: str) -> None:
        self.command = tuple(command)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"{' '.join(self.command)!r} exited {returncode}: {stderr.strip()}")


def _run_subprocess(
    args: list[str], *, timeout: float, binary: bool
) -> subprocess.CompletedProcess:
    """The real runner. Replaced with a fake in tests so command construction
    and error handling can be checked without a machine that has adb."""
    return subprocess.run(args, capture_output=True, timeout=timeout, text=not binary)


class AdbBridge:
    """One ADB connection to one MuMu instance.

    ``runner`` is the seam that makes this class testable without a device:
    it defaults to a thin wrapper around :func:`subprocess.run`, but any
    ``Callable[[list[str]], CompletedProcess]``-shaped object taking the same
    ``timeout``/``binary`` keywords can stand in for it. Every test in
    ``tests/test_mumu_adb.py`` that does not carry a ``skipif`` uses this to
    verify the exact argv this class builds, rather than exercising real adb.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        adb_path: str | Path | None = None,
        timeout: float = 10.0,
        runner: Callable[..., subprocess.CompletedProcess] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._explicit_adb_path = Path(adb_path) if adb_path is not None else None
        self._resolved_adb_path: Path | None = None
        self._run = runner or _run_subprocess

    @property
    def serial(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def adb_path(self) -> Path:
        """Resolved lazily, and only once, so constructing a bridge never
        touches the filesystem or PATH until a command actually needs to
        run -- that is what keeps a bridge cheap to construct in a test that
        never calls a method needing the real binary."""
        if self._explicit_adb_path is not None:
            return self._explicit_adb_path
        if self._resolved_adb_path is None:
            found = find_adb_binary()
            if found is None:
                raise AdbNotFoundError(
                    "no adb binary on PATH and none found at a known MuMu install "
                    "location; install Android platform-tools or pass "
                    "AdbBridge(adb_path=...) directly"
                )
            self._resolved_adb_path = found
        return self._resolved_adb_path

    def _invoke(self, args: list[str], *, binary: bool = False) -> subprocess.CompletedProcess:
        try:
            result = self._run(args, timeout=self.timeout, binary=binary)
        except subprocess.TimeoutExpired as exc:
            raise AdbTimeoutError(
                f"{' '.join(args)!r} did not return within {self.timeout}s"
            ) from exc
        if result.returncode != 0:
            stderr = result.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            raise AdbCommandError(args, result.returncode, stderr or "")
        return result

    def devices(self) -> tuple[AdbDevice, ...]:
        """Every device or emulator adb currently knows about, connected or
        not. Deliberately not filtered to just this bridge's own serial --
        :meth:`is_connected` does that -- because seeing every entry is what
        makes a stray "offline" or "unauthorized" line diagnosable."""
        result = self._invoke([str(self.adb_path), "devices"])
        devices: list[AdbDevice] = []
        for line in result.stdout.splitlines()[1:]:  # skip "List of devices attached"
            parts = line.split()
            if len(parts) >= 2:
                devices.append(AdbDevice(serial=parts[0], state=parts[1]))
        return tuple(devices)

    def is_connected(self) -> bool:
        return any(d.serial == self.serial and d.state == "device" for d in self.devices())

    def connect(self) -> None:
        """``adb connect host:port``.

        adb's own exit code for ``connect`` is not a reliable success signal
        across versions -- a refused connection is sometimes reported only in
        stdout text with the process still exiting 0. :meth:`is_connected` is
        checked afterward as the actual source of truth rather than trusting
        the return code alone.
        """
        self._invoke([str(self.adb_path), "connect", self.serial])
        if not self.is_connected():
            raise AdbCommandError(
                ["connect", self.serial],
                0,
                f"{self.serial} did not appear as 'device' in 'adb devices' after connect",
            )

    def shell(self, cmd: str) -> str:
        """Run ``cmd`` in the device's shell and return its stdout as text.

        Not used for screen capture -- ``adb shell`` allocates a pty-like
        channel on Windows that mangles binary data (see
        :mod:`cr_sim.mumu.capture`'s module docstring for the specific
        failure). Fine for the text commands this method is for.
        """
        result = self._invoke([str(self.adb_path), "-s", self.serial, "shell", cmd])
        return result.stdout

    def exec_out(self, cmd: str) -> bytes:
        """Run ``cmd`` via ``adb exec-out`` and return its stdout as raw
        bytes. Unlike :meth:`shell`, this does not allocate the pty-like
        channel that corrupts binary output on Windows, which is why
        :func:`cr_sim.mumu.capture.screencap` goes through this and not
        :meth:`shell`. The runner is called with ``binary=True`` so the
        underlying pipe is never opened in text mode -- decoding as text and
        re-encoding would reintroduce the exact corruption this method
        exists to avoid.
        """
        result = self._invoke(
            [str(self.adb_path), "-s", self.serial, "exec-out", cmd], binary=True
        )
        return result.stdout
