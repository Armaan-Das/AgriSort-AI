# AgriSort AI

A web application that analyses photos of apples and classifies their visible quality using computer vision and deep learning. It's a **software-only** academic engineering project, with no hardware or physical sorting.

**Current prototype (model v1):** classifies a single-apple photo as **Good** or **Rotten**. A **Damaged** class and detection of individual apples in multi-apple photos are planned.

## Features

- **Analyse Apple:** upload a JPG or PNG to get the predicted class (green ✔ Good / red ✖ Rotten), the confidence score, both class probabilities and the inference time. It also shows OpenCV image measurements (brightness, contrast, sharpness, colour channels, edge map).
- **History:** every saved inspection, stored in SQLite, with CSV export.
- **Dashboard:** statistics and Plotly charts calculated only from saved inspections, plus a summary of the held-out test results.
- **Model Information:** training configuration, dataset split, test metrics, confusion matrix, learning curves and limitations. Everything on this page is read from the evaluation report files.

## Technology stack

| Layer | Technology |
|---|---|
| Web interface | Streamlit 1.67 |
| Language | Python 3.12 |
| Deep learning | PyTorch 2.15 (Apple MPS GPU support) |
| Model | Ultralytics YOLO11n-cls (8.4.218), transfer learning from ImageNet |
| Image processing | OpenCV |
| Database | SQLite (built into Python) |
| Charts | Plotly |
| Testing | pytest |

## Installation (macOS, Apple Silicon)

```bash
cd ~/Developer/AgriSort-AI
python3.12 -m venv .venv          # only if .venv does not exist yet
source .venv/bin/activate
pip install -r requirements.txt
```

**Model weights:** the trained model `models/agrisort_good_rotten_v1.pt` (3.2 MB) is **not stored in git**. Copy it into `models/`, or recreate it with the pipeline below. Without it, the app still runs, but it shows a "model unavailable" message and makes no predictions.

## Launch the application

Run this from the project root, so Streamlit picks up the theme in `.streamlit/config.toml`:

```bash
source .venv/bin/activate
streamlit run app/main.py
```

Open http://localhost:8501. Press `Ctrl+C` in the terminal to stop it. If port 8501 is in use, add `--server.port 8503`.

**Labelling tool** (a separate app):

```bash
streamlit run labelling/label_tool.py --server.port 8502
```

## Run the tests

```bash
python -m pytest tests -q
```

The tests use a temporary database and never change `data/agrisort.db`. They include the full upload → prediction → save → History → Dashboard workflow.

## Dataset and attribution

Apple images come from **FruitNet: Indian Fruits Dataset with quality** (V. Meshram & K. Patil, *Data in Brief* 40, 2022), Mendeley Data, DOI [10.17632/b6fftwbr2v.3](https://data.mendeley.com/datasets/b6fftwbr2v/3), licensed under **CC BY 4.0**. The AgriSort AI team **relabelled the images by hand** into Good / Damaged / Rotten / Exclude (FruitNet's original labels were Good / Bad / Mixed).

| | Good | Rotten | Total |
|---|---|---|---|
| Train | 117 | 259 | 376 |
| Validation | 32 | 50 | 82 |
| Test (held out) | 27 | 53 | 80 |

The split is **group-aware**: burst photos and near-duplicates of the same apple are always kept in the same set, and the leakage check passed. The 65 Damaged labels are kept for future work. The full procedure, audit and labelling records are in [docs/DATASET.md](docs/DATASET.md). The images themselves aren't in git; the download and preparation steps are in that document.

## Model performance (held-out test set, evaluated once)

YOLO11n-cls, 224 px, 10 epochs, batch 16, class-weighted loss, trained on the Apple M4 GPU in 38 s.

| Metric | Value | 95% interval (resampling the 17 test groups) |
|---|---|---|
| Accuracy | **0.838** (67 / 80) | 0.754 – 0.921 |
| Balanced accuracy | **0.850** | 0.725 – 0.927 |
| Macro F1 | **0.828** | |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Good | 0.706 | 0.889 | 0.787 |
| Rotten | 0.935 | 0.811 | 0.869 |

Confusion matrix (rows = true class): Good → 24 correct / 3 predicted Rotten. Rotten → 43 correct / 10 predicted Good.

Median inference time is 3.6 ms per image on the CPU. The full report, with learning curves and error analysis, is in [reports/agrisort_good_rotten_v1/REPORT.md](reports/agrisort_good_rotten_v1/REPORT.md).

## Current limitations

1. **Two classes only.** Damaged apples are forced into Good or Rotten. Photos that aren't of apples still get a Good/Rotten answer, and multi-apple photos are classified as a whole.
2. **Small test set:** 80 images from only 17 independent groups, so the intervals above are wide.
3. **Camera and background bias:** most Rotten photos are single apples held in a hand against a wall, and 217 come from one phone. 6 of the 10 missed rotten test images show two apples on a plain surface, which suggests the model partly relies on the setting.
4. **Class imbalance (1 Good : 2.1 Rotten),** handled with verified class-weighted loss.
5. **Approximate grouping:** the same apple photographed more than 20 s apart can't be detected, so the results may be slightly optimistic.
6. **One annotator and one dataset,** not yet tested on new photos, other varieties or other lighting.
7. **The confidence score is not a guarantee of correctness, and the app is not a food-safety assessment.**

## Reproduce the pipeline

```bash
python scripts/fruitnet_extract.py                             # verify download + extract apples
python scripts/fruitnet_audit.py --session-gap 20 --threshold 8 # duplicates, near-duplicates, groups
streamlit run labelling/label_tool.py --server.port 8502       # manual labelling + review
python scripts/merge_labels.py
python scripts/build_yolo_dataset.py --classes Good Rotten     # group-aware 70/15/15 split
python scripts/verify_split.py                                 # independent leakage check
python scripts/train_classifier.py --name <new_run_name>       # YOLO11n-cls training
python scripts/evaluate_classifier.py --weights models/<new_run_name>.pt
```

## Project structure

```
AgriSort-AI/
├── app/                   # Streamlit website
│   ├── main.py            # entry point: page setup, navigation, sidebar
│   ├── views/             # Home, Analyse Apple, Dashboard, History, Model Information, About
│   ├── components/ui.py   # shared styling and UI elements
│   └── core/              # config, database, image utilities, classifier, model report loader
├── labelling/             # standalone labelling tool
├── scripts/               # dataset pipeline, training and evaluation
├── data/labels/           # human annotations and review log (tracked)
├── data/{raw,processed,downloads,interim,backups}/   # local only, not in git
├── models/                # trained weights (local only, not in git)
├── reports/               # training and evaluation results (tracked)
├── docs/DATASET.md        # dataset card
├── tests/                 # automated tests
├── .streamlit/config.toml # theme
├── requirements.txt
└── CLAUDE.md              # project rules for AI-assisted development
```

## Roadmap

1. ~~Web application, OpenCV analysis, SQLite history, dashboard~~
2. ~~FruitNet dataset audit and manual relabelling (round 1)~~
3. ~~Train and evaluate YOLO11n-cls (Good vs Rotten)~~
4. ~~Integrate the model into the website, with genuine metrics~~
5. Add the Damaged class (more labelled data) and test on new photos
6. Detect and grade individual apples in multi-apple photos
