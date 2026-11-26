# Licensing and attribution notice

AgriSort AI
Copyright (C) 2026 AgriSort AI Team

This repository contains material under **three different licences**. The table below says which licence applies to which part. If a file is not listed, the project licence (AGPL-3.0-or-later) applies.

| Part of the repository | Licence | Notes |
|---|---|---|
| Source code: `app/`, `scripts/`, `labelling/`, `tests/`, configuration files | **GNU AGPL-3.0-or-later**, see [`LICENSE`](LICENSE) | Uses the Ultralytics YOLO library, which is licensed under AGPL-3.0. |
| Trained model: `models/agrisort_good_rotten_v1.pt` | **GNU AGPL-3.0-or-later** | Fine-tuned from Ultralytics' pretrained `yolo11n-cls.pt` weights. Ultralytics states that models trained with its software fall under AGPL-3.0. The model was trained on FruitNet images (CC BY 4.0), so the FruitNet attribution below also applies. |
| Human annotations: `data/labels/*.csv` | **CC BY 4.0**, see [`data/labels/LICENSE.md`](data/labels/LICENSE.md) | Original labelling work by the AgriSort AI Team. The labels describe FruitNet images, which are not included. |
| FruitNet images inside `reports/**/*.jpg` and `docs/screenshots/*.png` | **CC BY 4.0**, © V. Meshram and K. Patil | Third-party images. They are **not** covered by the AGPL. See the attribution below. |
| Documentation, reports and other images | **GNU AGPL-3.0-or-later** | Except for the FruitNet image content noted above. |

## FruitNet dataset attribution (CC BY 4.0)

> Apple images from **FruitNet: Indian Fruits Dataset with quality (Good, Bad & Mixed quality)**, version 3, by **Vishal Meshram and Kailas Patil**, Mendeley Data, 2022. DOI: [10.17632/b6fftwbr2v.3](https://doi.org/10.17632/b6fftwbr2v.3). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
>
> Related article: V. Meshram, K. Patil, *FruitNet: Indian fruits image dataset with quality for machine learning applications*, Data in Brief 40 (2022) 107686. https://doi.org/10.1016/j.dib.2021.107686

**Changes made by the AgriSort AI Team.** We extracted the apple images only. A person relabelled them into Good / Damaged / Rotten / Exclude (FruitNet's original labels were Good / Bad / Mixed). We used them to train and evaluate a classifier. Some images appear resized, cropped, tiled into grids, and annotated with predicted and true labels:
- in the Ultralytics batch previews (`reports/agrisort_good_rotten_v1/ultralytics/*_batch*.jpg`)
- in the error-analysis grid (`reports/agrisort_good_rotten_v1/test_evaluation/misclassified.jpg`)
- in the application screenshots (`docs/screenshots/`)

**This repository does not redistribute the FruitNet dataset.** To reproduce training, download it from the original source; see [`docs/DATASET.md`](docs/DATASET.md).

The FruitNet authors do not endorse this project.

## Third-party software (installed with pip, not included in this repository)

| Package | Licence |
|---|---|
| Ultralytics (YOLO11) | AGPL-3.0 |
| PyTorch | BSD-3-Clause and other permissive licences (see package) |
| torchvision | BSD-3-Clause |
| Streamlit | Apache-2.0 |
| OpenCV (`opencv-python`) | Apache-2.0 |
| Plotly | MIT |
| pandas | BSD-3-Clause |
| NumPy | BSD-3-Clause (and other permissive licences, see package) |
| Pillow | MIT-CMU |
| pytest (development only) | MIT |

All of these licences are compatible with distributing this project under AGPL-3.0-or-later.

## What the AGPL means in practice

- Anyone may use, study, modify and share this project, including the trained model.
- If you distribute a modified version, **or run a modified version as a network service** (for example a hosted Streamlit app), you must make its complete source code available to its users under the same licence.
- The software and model are provided **without any warranty**. Predictions are not a food-safety assessment.
