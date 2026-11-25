# AgriSort AI

AI-powered web application that analyses images of apples and classifies their visible quality as **Good**, **Damaged** or **Rotten**.

This is a software-only academic engineering project. There's no hardware or physical sorting.

## Current status

| Component | Status |
|---|---|
| Streamlit website (Home, Analyse Apple, Dashboard, History, About) | Working |
| OpenCV image analysis (brightness, contrast, sharpness, colour, edge map) | Working |
| SQLite history storage | Working |
| Plotly dashboard (built from real saved records only) | Working |
| Apple image dataset (FruitNet, CC BY 4.0) | Downloaded and audited, manual labelling in progress (see [docs/DATASET.md](docs/DATASET.md)) |
| YOLO classification model | Prototype v1 trained (Good vs Rotten, YOLO11n-cls): test balanced accuracy 0.850, see [report](reports/agrisort_good_rotten_v1/REPORT.md). **Not yet connected to the website.** |

The app doesn't show quality predictions yet. A first Good-vs-Rotten model has been trained and evaluated, and connecting it to the Analyse page is the next step. Damaged is planned for the final model.

## Tech stack

Python 3.12 · Streamlit · PyTorch · Ultralytics YOLO · OpenCV · SQLite · Plotly

## Setup (macOS, Apple Silicon)

```bash
cd ~/Developer/AgriSort-AI
python3.12 -m venv .venv          # only if .venv does not exist yet
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app

Run this from the project root so Streamlit picks up the theme in `.streamlit/config.toml`:

```bash
source .venv/bin/activate
streamlit run app/main.py
```

Then open http://localhost:8501 in your browser. Press `Ctrl+C` in the terminal to stop the app.

## Run the tests

```bash
python -m pytest tests -q
```

The tests use a temporary database and never modify `data/agrisort.db`.

## Project structure

```
AgriSort-AI/
├── app/
│   ├── main.py            # entry point: page setup, navigation, sidebar
│   ├── views/             # one file per page
│   ├── components/ui.py   # shared styling and UI elements
│   └── core/              # config, database, OpenCV utilities, classifier (placeholder)
├── data/                  # raw/ and processed/ datasets (later), agrisort.db
├── models/                # trained weights (later)
├── notebooks/             # experiments
├── scripts/               # training / evaluation scripts (later)
├── tests/                 # automated tests
├── .streamlit/config.toml # theme
├── requirements.txt
└── CLAUDE.md              # project rules for AI-assisted development
```

## Roadmap

1. ~~Web application, OpenCV analysis, SQLite, dashboard~~
2. Collect and label an apple image dataset (Good / Damaged / Rotten)
3. Train a YOLO classification model (on the Apple MPS GPU) and evaluate it on a held-out test set
4. Integrate the trained model into the Analyse Apple page
5. Report real evaluation metrics on the dashboard
