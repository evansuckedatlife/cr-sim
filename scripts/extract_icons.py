"""Pull card artwork out of the APK and shrink it for the replay viewer.

The shipped PNGs are full-resolution card art -- 111 files, ~15MB, several
hundred KB each. The viewer draws them at roughly 20 pixels across inside a
circle, so they are downscaled hard here rather than at render time: embedding
them raw would make a single replay bigger than the entire simulator.

Usage:
    python scripts/extract_icons.py <apk-or-directory> [--size 64]
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_PREFIXES = (
    "assets/image/chr/",
    "assets/image/chr_evolution/",
    "assets/image/chr_champions/",
    "assets/image/chr_goblin_faction/",
    "assets/image/chr_support_cards/",
)


def apk_candidates(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.iterdir() if p.suffix.lower() in {".apk", ".zip"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path, help="APK file, or a directory of split APKs")
    ap.add_argument("--out", type=Path, default=Path("data_cache/icons"))
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument(
        "--no-crop",
        action="store_true",
        help="keep the original transparent padding instead of trimming it",
    )
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required: python -m pip install pillow", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    raw_bytes = out_bytes = 0

    for apk in apk_candidates(args.target):
        try:
            zf = zipfile.ZipFile(apk)
        except zipfile.BadZipFile:
            continue
        with zf:
            members = [n for n in zf.namelist() if n.startswith(IMAGE_PREFIXES)]
            if not members:
                continue
            print(f"{apk.name}: {len(members)} images")
            for member in members:
                data = zf.read(member)
                raw_bytes += len(data)
                try:
                    image = Image.open(io.BytesIO(data)).convert("RGBA")
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! {member}: {exc}")
                    continue
                # Card art carries a lot of empty margin -- Knight's character
                # fills only 59% of its PNG. Drawn inside a circle that margin
                # is doubly wasted, so trim to the visible pixels and pad back
                # to a square, which keeps the aspect ratio while letting the
                # character fill the frame.
                if not args.no_crop:
                    bbox = image.getbbox()
                    if bbox:
                        image = image.crop(bbox)
                    side = max(image.size)
                    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
                    square.paste(
                        image, ((side - image.width) // 2, (side - image.height) // 2)
                    )
                    image = square
                image.thumbnail((args.size, args.size), Image.LANCZOS)
                buffer = io.BytesIO()
                # optimize + max compression: these are tiny and embedded as
                # base64, where every byte costs ~1.33 in the final HTML.
                image.save(buffer, format="PNG", optimize=True, compress_level=9)
                payload = buffer.getvalue()
                out_bytes += len(payload)
                (args.out / Path(member).name).write_bytes(payload)
                written += 1

    if not written:
        print(f"no images found under {IMAGE_PREFIXES} in {args.target}", file=sys.stderr)
        return 1
    print(
        f"wrote {written} icons to {args.out} "
        f"({raw_bytes / 1e6:.1f}MB -> {out_bytes / 1e3:.0f}KB at {args.size}px)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
