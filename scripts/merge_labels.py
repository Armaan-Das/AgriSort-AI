"""Step 4 - Merge the label files produced by the labelling tool.

Reads every data/labels/fruitnet_labels_*.csv (the 'all' file and batch files 1-6)
and writes data/labels/fruitnet_labels_merged.csv.

If two files give the SAME image DIFFERENT labels, it is a conflict: the image is
left out of the merged file and listed in data/labels/fruitnet_label_conflicts.csv
so a person can resolve it. Nothing is decided automatically.

Usage:
    python scripts/merge_labels.py
"""

from __future__ import annotations

import collections

from dataset_common import (
    ALL_LABELS,
    LABEL_FIELDS,
    LABELS_DIR,
    MERGED_LABELS_CSV,
    read_csv,
    write_csv_atomic,
)

CONFLICTS_CSV = LABELS_DIR / "fruitnet_label_conflicts.csv"


def merge() -> tuple[list[dict], list[dict], dict]:
    files = sorted(f for f in LABELS_DIR.glob("fruitnet_labels_*.csv") if f != MERGED_LABELS_CSV)
    by_image: dict[str, list[dict]] = collections.defaultdict(list)
    for f in files:
        for r in read_csv(f):
            if r["label"] not in ALL_LABELS:
                raise SystemExit(f"Unknown label {r['label']!r} in {f.name}")
            by_image[r["image_id"]].append({**r, "file": f.name})

    merged, conflicts = [], []
    for image_id, rows in sorted(by_image.items()):
        if len({r["label"] for r in rows}) == 1:
            merged.append(max(rows, key=lambda r: r["labelled_at"]))
        else:
            conflicts.extend(rows)
    info = {"files": [f.name for f in files], "images": len(by_image),
            "merged": len(merged), "conflicting_images": len({r["image_id"] for r in conflicts})}
    return merged, conflicts, info


def main() -> None:
    merged, conflicts, info = merge()
    write_csv_atomic(MERGED_LABELS_CSV, merged, LABEL_FIELDS)
    write_csv_atomic(CONFLICTS_CSV, conflicts, LABEL_FIELDS + ["file"])
    print(f"Label files read : {', '.join(info['files']) or '(none)'}")
    print(f"Images labelled  : {info['images']}")
    print(f"Merged           : {info['merged']}  -> {MERGED_LABELS_CSV.name}")
    print(f"Conflicts        : {info['conflicting_images']}  -> {CONFLICTS_CSV.name}")
    counts = collections.Counter(r["label"] for r in merged)
    for name in ALL_LABELS:
        print(f"  {name:8s} {counts.get(name, 0)}")


if __name__ == "__main__":
    main()
