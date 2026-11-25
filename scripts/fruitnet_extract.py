"""Step 1 - Verify the FruitNet download and extract ONLY the apple images.

Source : FruitNet: Indian Fruits Dataset with quality (Good, Bad & Mixed quality)
         Meshram & Patil, Mendeley Data v3, DOI 10.17632/b6fftwbr2v.3, CC BY 4.0
Input  : data/downloads/FruitNetDataset.zip   (download command in docs/DATASET.md)
Output : data/raw/fruitnet/{Good,Bad,Mixed}/<original file name>
         data/raw/fruitnet/extract_manifest.csv  (original zip path -> extracted path)

Usage:
    python scripts/fruitnet_extract.py            # verify checksum, then extract
    python scripts/fruitnet_extract.py --list     # only show the zip's folder layout
"""

from __future__ import annotations

import argparse
import collections
import shutil
import sys
import zipfile
from pathlib import PurePosixPath

from dataset_common import (
    FRUITNET_ZIP_SHA256,
    IMAGE_EXTENSIONS,
    RAW_DIR,
    ZIP_PATH,
    sha256_file,
    write_csv_atomic,
)


def quality_of(parts: list[str]) -> str | None:
    """Map a folder name such as 'Bad Quality_Fruits' to Good / Bad / Mixed."""
    for p in parts:
        low = p.lower()
        if "good" in low:
            return "Good"
        if "bad" in low:
            return "Bad"
        if "mix" in low:
            return "Mixed"
    return None


def is_apple(parts: list[str]) -> bool:
    return any("apple" in p.lower() for p in parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show folder layout and exit")
    ap.add_argument("--skip-checksum", action="store_true", help="skip SHA-256 verification")
    args = ap.parse_args()

    if not ZIP_PATH.exists():
        sys.exit(f"Not found: {ZIP_PATH}\nDownload it first (see docs/DATASET.md).")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]

        if args.list:
            folders = collections.Counter(str(PurePosixPath(i.filename).parent) for i in infos)
            for folder, n in sorted(folders.items()):
                print(f"{n:6d}  {folder}")
            return

        if not args.skip_checksum:
            print("Verifying SHA-256 checksum (takes ~10 s)...")
            digest = sha256_file(ZIP_PATH)
            if digest != FRUITNET_ZIP_SHA256:
                sys.exit(f"Checksum mismatch!\n expected {FRUITNET_ZIP_SHA256}\n got      {digest}")
            print("Checksum OK - file matches the one published on Mendeley Data.")

        selected, skipped_apple = [], []
        for info in infos:
            path = PurePosixPath(info.filename)
            parts = list(path.parts[:-1])
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not is_apple(parts):
                continue
            # Safety: never trust archive paths (no absolute paths or '..').
            if path.is_absolute() or ".." in path.parts:
                skipped_apple.append((info.filename, "unsafe path"))
                continue
            quality = quality_of(parts)
            if quality is None:
                skipped_apple.append((info.filename, "unknown quality folder"))
                continue
            selected.append((info, quality))

        if RAW_DIR.exists():
            print(f"Removing previous extraction in {RAW_DIR} ...")
            shutil.rmtree(RAW_DIR)

        manifest, used = [], set()
        for info, quality in selected:
            name = PurePosixPath(info.filename).name
            rel = f"{quality}/{name}"
            n = 1
            while rel in used:  # avoid overwriting files that share a name
                n += 1
                rel = f"{quality}/{PurePosixPath(name).stem}__{n}{PurePosixPath(name).suffix}"
            used.add(rel)
            dest = RAW_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            manifest.append({"image_id": rel, "source_quality": quality, "zip_path": info.filename,
                             "bytes": info.file_size})

    write_csv_atomic(RAW_DIR / "extract_manifest.csv", manifest,
                     ["image_id", "source_quality", "zip_path", "bytes"])

    counts = collections.Counter(m["source_quality"] for m in manifest)
    print(f"\nExtracted {len(manifest)} apple images into {RAW_DIR}")
    for q in ("Good", "Bad", "Mixed"):
        print(f"  {q:6s}: {counts.get(q, 0)}")
    if skipped_apple:
        print(f"\nSkipped {len(skipped_apple)} apple files:")
        for name, why in skipped_apple[:20]:
            print(f"  [{why}] {name}")


if __name__ == "__main__":
    main()
