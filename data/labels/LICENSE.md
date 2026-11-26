# Licence for the AgriSort AI annotations

The CSV files in this folder are the AgriSort AI Team's human annotations of FruitNet apple images:
- quality labels
- review decisions
- the list of flagged images

Copyright (C) 2026 AgriSort AI Team.

They are licensed under the **Creative Commons Attribution 4.0 International licence (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/

**Suggested attribution:**

> Apple quality annotations by the AgriSort AI Team (2026), https://github.com/Armaan-Das/AgriSort-AI, licensed under CC BY 4.0. The annotations refer to images from FruitNet (Meshram & Patil, 2022, DOI 10.17632/b6fftwbr2v.3, CC BY 4.0).

**The images themselves are not included.** They belong to the FruitNet authors and are available from the FruitNet record on Mendeley Data under CC BY 4.0. If you use the images together with these labels, you must credit FruitNet as well. See [`NOTICE.md`](../../NOTICE.md).

## Columns

| File | Columns |
|---|---|
| `fruitnet_labels_all.csv`, `fruitnet_labels_merged.csv` | `image_id` (FruitNet folder/file), `label` (Good / Damaged / Rotten / Exclude), `source_quality` (original FruitNet folder), `annotator` (initials), `labelled_at` (local time) |
| `review_round1_flagged.csv` | Images flagged for a second human look, with the reason |
| `review_round1_decisions.csv` | Each review decision, with the original and reviewed labels |
| `fruitnet_label_conflicts.csv` | Disagreements found during merging (currently none) |
