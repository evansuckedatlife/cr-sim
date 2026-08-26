"""Pull ``assets/csv_logic`` out of a Clash Royale APK (or split APK set) and
decode every file into ``data_cache/csv_logic``.

Usage:
    python scripts/extract_apk.py <apk-or-directory> [--out data_cache/csv_logic]
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cr_sim.data.decode import DecodeError, decode_bytes

ASSET_PREFIXES = ("assets/csv_logic/", "assets/tilemaps/")


def apk_candidates(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.iterdir() if p.suffix.lower() in {".apk", ".zip"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path, help="APK file, or a directory of split APKs")
    ap.add_argument("--out", type=Path, default=Path("data_cache"))
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    written = failed = 0
    schemes: dict[str, int] = {}
    source_apk = None

    for apk in apk_candidates(args.target):
        try:
            zf = zipfile.ZipFile(apk)
        except zipfile.BadZipFile:
            continue
        with zf:
            members = [
                n
                for n in zf.namelist()
                if n.startswith(ASSET_PREFIXES) and not n.endswith("/")
            ]
            if not members:
                continue
            source_apk = apk
            print(f"{apk.name}: {len(members)} logic files")
            for member in members:
                name = member[len("assets/") :]
                try:
                    decoded = decode_bytes(zf.read(member), name=name)
                except DecodeError as exc:
                    print(f"  !! {name}: {exc}")
                    failed += 1
                    continue
                dest = out / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(decoded.text, encoding="utf-8", newline="")
                schemes[decoded.scheme] = schemes.get(decoded.scheme, 0) + 1
                written += 1

    if source_apk is None:
        print(f"none of {ASSET_PREFIXES} found under {args.target}", file=sys.stderr)
        return 1

    (out / "_PROVENANCE.txt").write_text(
        f"extracted from {source_apk}\nfiles: {written}\nschemes: {schemes}\n",
        encoding="utf-8",
    )
    print(f"wrote {written} files to {out} ({schemes}), {failed} failed")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
