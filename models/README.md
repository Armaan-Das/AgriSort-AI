# Model card: `agrisort_good_rotten_v1.pt`

| Item | Value |
|---|---|
| File | `models/agrisort_good_rotten_v1.pt`, 3,202,495 bytes |
| SHA-256 | `29f7f67e7a935c38ba246b4cdb14ecf6636a907f9e91c5b156c1ce2b641cc7c0` |
| Task | Single-image classification of apple quality: **Good** or **Rotten** |
| Architecture | Ultralytics YOLO11n-cls, 1,533,666 parameters with the 2-class head, stored as FP16 |
| Starting point | Ultralytics' ImageNet-pretrained `yolo11n-cls.pt` (transfer learning) |
| Training data | FruitNet apple images (CC BY 4.0), relabelled by people, group-aware split. See [`docs/DATASET.md`](../docs/DATASET.md) |
| Training | 10 epochs, image size 224, batch 16, class-weighted loss (`cls_pw=1.0`), seed 42, Apple M4 GPU (MPS), Ultralytics 8.4.218 |
| Held-out test result | Accuracy 0.838, balanced accuracy 0.850, macro F1 0.828 on 80 images from 17 independent groups. Full report: [`reports/agrisort_good_rotten_v1/REPORT.md`](../reports/agrisort_good_rotten_v1/REPORT.md) |
| Licence | AGPL-3.0-or-later (see [`NOTICE.md`](../NOTICE.md)) |

## Intended use

An academic prototype that demonstrates visual quality grading of **one apple per photo**. It is **not** a food-safety assessment. Known limitations are listed in the report and in the main README. In short:
- There are only two classes, so damaged apples are forced into Good or Rotten.
- The test set is small.
- The model probably relies partly on photo setting and camera.
- It has not been tested on new photos.

## How this file was prepared for publication

The trained checkpoint (`best.pt`, epoch 8) was copied to `models/` after training. Before publication, **only the training metadata** stored inside the file was edited:
- In `train_args` (`model`, `data`, `project`), `git.root` and `model.pt_path`, the local absolute folder path was replaced with `<project-root>`.

**The network itself was not changed.** This was checked as follows:

| Check | Result |
|---|---|
| All 236 tensors in the state dict (1,543,865 values) compared with the original | Identical (same dtype, shape and values) |
| Predictions on all 80 held-out test images, original vs published file | Same class 80/80, maximum probability difference **0** |
| Published file vs the predictions recorded in `test_evaluation/predictions.csv` | 80/80 match |

The original, unedited checkpoint has SHA-256 `aeaf3f78cf15ece7d2fda893b84816e34e5fe222964da8d4f3399859f3b66713`. It is kept privately by the team and is not distributed.

## Verify your copy

```bash
# macOS / Linux
shasum -a 256 models/agrisort_good_rotten_v1.pt
# Windows (PowerShell)
Get-FileHash models\agrisort_good_rotten_v1.pt -Algorithm SHA256
```
