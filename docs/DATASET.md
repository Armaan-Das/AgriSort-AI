# AgriSort AI dataset card: FruitNet apples (manually relabelled)

*Last updated: 25 November 2026. Status: images downloaded, verified and audited. Human labelling not started yet. No model has been trained.*

## 1. Source and licence

| Item | Value |
|---|---|
| Dataset | FruitNet: Indian Fruits Dataset with quality (Good, Bad & Mixed quality), version 3 |
| Authors | Vishal Meshram, Kailas Patil |
| Repository | Mendeley Data: https://data.mendeley.com/datasets/b6fftwbr2v/3 |
| DOI | 10.17632/b6fftwbr2v.3 |
| Paper | V. Meshram, K. Patil, *FruitNet: Indian fruits image dataset with quality for machine learning applications*, Data in Brief 40 (2022) 107686. https://doi.org/10.1016/j.dib.2021.107686 |
| Licence | **CC BY 4.0** (Creative Commons Attribution 4.0 International), as stated on the Mendeley Data record |
| Provenance | The authors photographed the fruit themselves on mobile phones. These are not images collected from the web. |

**Required attribution.** Use this text on any page, report or slide that uses the data:

> Apple images from FruitNet (Meshram & Patil, 2022), Mendeley Data, DOI 10.17632/b6fftwbr2v.3, licensed under CC BY 4.0. Images were relabelled by the AgriSort AI team into Good / Damaged / Rotten; the original labels were Good / Bad / Mixed.

CC BY 4.0 requires attribution and a statement that changes were made. Relabelling counts as a change, which is why the text above mentions it.

**Why not Healthy-Defective Fruits?** That dataset's paper (VISAPP 2023, section 3.2) says its real images *"were collected from various sources online"*. The authors therefore couldn't grant reuse rights for those photos, so it was rejected.

## 2. Download and verification

| Item | Value |
|---|---|
| File | `FruitNetDataset.zip`, 3,253,463,240 bytes |
| URL | https://data.mendeley.com/public-files/datasets/b6fftwbr2v/files/532e56bf-d855-4afd-86fd-2c8c425eb32c/file_downloaded |
| SHA-256 (published) | `b84851e0c9101156bc537a7adfc4528289fa063d8a9a029ea4a6dd713cf5e53d` |
| Verification | ✅ The downloaded file's checksum matches the published value. |
| Stored at | `data/downloads/FruitNetDataset.zip` (not tracked by git). It can be deleted once extracted. |

Only the apple folders were extracted. Banana, guava, lime, orange and pomegranate were skipped.

## 3. Extracted apple images (actual counts)

| FruitNet folder (in zip) | Saved as | Images |
|---|---|---|
| `Good Quality_Fruits/Apple_Good` | `data/raw/fruitnet/Good/` | 1,149 |
| `Bad Quality_Fruits/Apple_Bad` | `data/raw/fruitnet/Bad/` | 1,141 |
| `Mixed Qualit_Fruits/Apple` | `data/raw/fruitnet/Mixed/` | 113 |
| **Total** | | **2,403** (45 MB) |

These counts match Table 4 of the FruitNet paper (1149 / 1141 / 113). `data/raw/fruitnet/extract_manifest.csv` maps every file back to its original path in the zip.

- **Image sizes:** 1,634 at 256×256, 630 at 192×256, 138 at 256×192, and 1 at 144×256. All are JPEG.
- **Cameras (from EXIF):** ZUK Z2 Plus 891, realme 5 Pro 403, Samsung SM-G950F 321, iPhone SE 309, iPhone 6 109. The other 370 have no EXIF camera field.

## 4. Audit results

Run with `python scripts/fruitnet_audit.py` (default settings, seed 42).

| Check | Result |
|---|---|
| Corrupted or unreadable images (Pillow verify + OpenCV decode) | **0** |
| Exact duplicates (identical bytes, SHA-256) | **0** |
| Near-duplicate pairs (perceptual hash, Hamming distance ≤ 6 of 64 bits) | **3,126** |
| Near-duplicates split across Good and Bad folders (label conflict) | **0** |
| Images with a capture time | 2,403 / 2,403 (2,033 from EXIF, 370 from the file name) |
| Capture bursts (photos less than 10 s apart) | 208 |
| Leakage groups (bursts + near-duplicates combined) | **199**: Good 158 (median size 4), Bad 33 (median 25, max 154), Mixed 8 |
| Groups containing more than one source folder | 0 |

Detailed outputs are in `data/interim/fruitnet_audit/`: `images.csv`, `near_duplicate_pairs.csv`, `label_conflicts.csv` and `summary.json`.

### Findings from manually inspecting contact sheets

1. **Burst photography.** Each apple was photographed 4–16 times within seconds, giving near-identical images. A random split would leak the same apple into the test set and inflate accuracy. This is handled by the group-aware split described in section 6.
2. **The "Bad" folder mixes two conditions.** It contains clearly rotten apples (brown, soft, mouldy) and physically damaged apples (dents, cuts, cracks) with no further label. That's why every Bad image needs a human decision between Damaged and Rotten.
3. **Some Bad images show 2–3 apples.** If those apples are in different conditions, the image should be marked Exclude.
4. **The "Mixed" folder shows piles of several apples on grass.** Nearly all of these are Exclude under our guidelines, and labelling them is optional.
5. **Possible setting bias.** Many Bad photos are of red apples held in a hand against a wall, while part of the Good folder is green apples on floors or a dark cloth. Other Good photos, from the 2020 sessions, are also hand-held against walls, so the overlap is partial. A model could still pick up on background or apple variety instead of defects. Test results should be interpreted with this in mind, and the model should be checked on new photos before we claim it generalises.
6. **The near-duplicate check can't catch every repeat.** It doesn't detect the same apple photographed on a different day from a very different angle. Capture-burst grouping is the best proxy available.

## 5. Class mapping and labelling guidelines

FruitNet labels are only used as hints. **Every Damaged and Rotten label is assigned by a human** using `labelling/label_tool.py`.

| AgriSort class | Definition |
|---|---|
| **Good** | No significant visible damage or decay. |
| **Damaged** | Visible bruises, cuts, cracks or physical damage, without clear signs of decay. |
| **Rotten** | Visible signs of decay, mould or decomposition. |
| **Exclude** | Ambiguous images, several apples in different conditions, or anything that can't be classified reliably. |

By default, Good images are also verified by a human. The `--use-source-good` option accepts FruitNet's own Good label for images that haven't been checked; anything used that way is reported as "unverified".

## 6. Preparation procedure (reproducible)

```bash
# 0. download (3.25 GB; resumable)
curl -L -C - -o data/downloads/FruitNetDataset.zip "<URL in section 2>"
# 1. verify checksum + extract apple images only
python scripts/fruitnet_extract.py
# 2. audit: corruption, duplicates, near-duplicates, capture bursts, batches
python scripts/fruitnet_audit.py
# 3. human labelling (see labelling/README.md)
streamlit run labelling/label_tool.py --server.port 8502
# 4. merge label files (batches 1-6 and/or "all"), report conflicts
python scripts/merge_labels.py
# 5. build the YOLO classification dataset
python scripts/build_yolo_dataset.py --balance
```

**Batches.** The audit assigns every image to one of 6 fixed batches that don't overlap. Whole groups go to the same batch, and batches are balanced within each source folder. Current batch sizes are 370 / 410 / 397 / 482 / 357 / 387.

**Splitting** (`build_yolo_dataset.py`):
- Uses only human labels of Good, Damaged or Rotten. Exclude labels, unreadable files and duplicate copies are dropped.
- Optional balancing (`--balance`) caps every class at the size of the smallest class, picking whole groups at random with a fixed seed.
- The split is **70 / 15 / 15**, stratified by class and **group-aware**: every leakage group goes entirely into train, val or test, never more than one. `summary.json` reports `groups_split_across_sets`, which must be 0.
- Because the Bad folder has only 33 groups, the achieved proportions are approximate and val/test contain only a few distinct capture sessions.

**Output** (Ultralytics YOLO classification layout):

```
data/processed/fruitnet_cls/
├── train/{Good,Damaged,Rotten}/
├── val/{Good,Damaged,Rotten}/
├── test/{Good,Damaged,Rotten}/
├── manifest.csv     # image, label, split, group, human_verified
└── summary.json     # counts per split and class, settings used
```

## 7. Current status

| Stage | Status |
|---|---|
| Download + checksum | ✅ Done |
| Apple extraction (2,403 images) | ✅ Done |
| Audit | ✅ Done (results above) |
| Labelling tool | ✅ Ready and tested |
| Human labels | ✅ Round 1 done: 730 images (see section 8) |
| `data/processed/fruitnet_cls/` | ⏳ Not built yet (awaiting approval of the training strategy) |
| Model training | ⛔ Not started (awaiting approval) |

## 8. Labelling round 1: assessment (25 November 2026)

Annotator AD labelled 730 images between 21:36 and 22:03. All rows were saved and validated: no duplicates, no unknown image IDs, no invalid labels and **0 merge conflicts**. A checksummed backup is in `data/backups/labels_20261125-220522/` (not tracked by git).

**Labels compared with the original FruitNet folders:**

| Label | From FruitNet "Good" folder | From FruitNet "Bad" folder | Total | Leakage groups | Capture sessions |
|---|---|---|---|---|---|
| Good | 169 | 0 | **169** | 65 | 68 |
| Damaged | 58 | 12 | **70** | 41 | 46 |
| Rotten | 86 | 287 | **373** | 68 | 75 |
| Exclude | 114 | 4 | 118 | 46 | 52 |

Coverage so far: 427 of 1,149 Good-folder images, 303 of 1,141 Bad-folder images, and 0 of 113 Mixed images.

**Findings:**
- **Clearly damaged apples are rare in FruitNet.** Only 12 Bad-folder images were labelled Damaged (6 groups, 6 of the images being one apple). That's about 4% of the Bad images reviewed.
- **144 Good-folder images were labelled Damaged or Rotten.** In a thumbnail review, most of them showed no clear defect: they looked like normal green or yellow apples with russet or speckles. They're listed for an optional human re-check in `data/interim/review_round1_flagged.csv`. **No labels were changed automatically.**
- The largest group (group 65, 226 images in total, from the green-apple sessions) contains 79 non-Exclude labels across three classes. Group-aware splitting keeps all of it in one split.

**Simulated group-aware 70/15/15 split** (all usable images, no balancing; 0 groups split across sets):

| Option | Train | Val | Test |
|---|---|---|---|
| 3-class (Good / Damaged / Rotten) | 117 / 48 / 264 | 29 / 12 / 51 | 23 / 10 / 58 |
| 2-class (Good / Rotten) | 117 / 262 | 26 / 56 | 26 / 55 |
