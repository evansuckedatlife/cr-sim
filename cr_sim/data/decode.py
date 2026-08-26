"""Decompressor for Supercell's packed ``csv_logic`` assets.

Files shipped inside the APK are compressed with one of three schemes, and the
first bytes tell them apart:

``5d 00 00 xx 00``
    LZMA-alone with a **truncated header**.  The standard container is 13 bytes
    (5 property bytes + an 8-byte little-endian uncompressed size); Supercell
    writes only 4 bytes for the size, so the stream has to be re-assembled
    before :mod:`lzma` will touch it.

``SCLZ``
    LZHAM.  Rare in ``csv_logic`` but present in some builds.

anything else
    Already plaintext -- passed through untouched, which is what the public
    GitHub dumps contain.
"""

from __future__ import annotations

import lzma
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DecodeError", "DecodedFile", "decode_bytes", "decode_file", "sniff"]

_LZMA_PROPS_LEN = 5
_SUPERCELL_SIZE_LEN = 4
_SCLZ_MAGIC = b"SCLZ"


class DecodeError(RuntimeError):
    """Raised when a packed asset cannot be decompressed."""


@dataclass(frozen=True, slots=True)
class DecodedFile:
    name: str
    text: str
    scheme: str  # "lzma" | "lzham" | "plain"


def sniff(raw: bytes) -> str:
    """Identify the packing scheme of ``raw`` without decompressing it."""
    if raw[:4] == _SCLZ_MAGIC:
        return "lzham"
    # LZMA props byte for Supercell assets is always 0x5d (lc=3, lp=0, pb=2)
    # followed by a 4-byte little-endian dictionary size.
    if len(raw) > 9 and raw[0] == 0x5D and raw[4] == 0x00:
        return "lzma"
    return "plain"


def _decode_lzma(raw: bytes) -> bytes:
    """Rebuild a well-formed LZMA-alone stream and decompress it."""
    props = raw[:_LZMA_PROPS_LEN]
    size_field = raw[_LZMA_PROPS_LEN : _LZMA_PROPS_LEN + _SUPERCELL_SIZE_LEN]
    body = raw[_LZMA_PROPS_LEN + _SUPERCELL_SIZE_LEN :]
    uncompressed = int.from_bytes(size_field, "little")

    # Pad the 4-byte size out to the 8 bytes the LZMA-alone container expects.
    stream = props + size_field + b"\x00" * 4 + body
    try:
        data = lzma.decompress(stream, format=lzma.FORMAT_ALONE)
    except lzma.LZMAError as exc:
        # Some builds write 0xFFFFFFFF for "unknown size"; retry as a raw stream.
        try:
            decomp = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)
            data = decomp.decompress(props + b"\xff" * 8 + body)
        except lzma.LZMAError:
            raise DecodeError(f"LZMA stream is not decodable: {exc}") from exc

    if uncompressed not in (0xFFFFFFFF, 0) and len(data) != uncompressed:
        raise DecodeError(
            f"decompressed {len(data)} bytes but header declared {uncompressed}"
        )
    return data


def _decode_lzham(raw: bytes) -> bytes:
    try:
        import lzham  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise DecodeError(
            "asset is LZHAM-packed (SCLZ); install the 'lzham' package to read it"
        ) from exc
    dict_size = raw[4]
    uncompressed = int.from_bytes(raw[5:9], "little")
    return lzham.decompress(raw[9:], {"dict_size_log2": dict_size}, uncompressed)


def decode_bytes(raw: bytes, *, name: str = "<bytes>") -> DecodedFile:
    """Decompress ``raw`` and decode it as text."""
    scheme = sniff(raw)
    if scheme == "lzma":
        data = _decode_lzma(raw)
    elif scheme == "lzham":
        data = _decode_lzham(raw)
    else:
        data = raw
    return DecodedFile(name=name, text=data.decode("utf-8-sig", errors="replace"), scheme=scheme)


def decode_file(path: str | Path) -> DecodedFile:
    path = Path(path)
    return decode_bytes(path.read_bytes(), name=path.name)
