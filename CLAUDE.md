# CLAUDE.md: AgriSort AI

Guidance for AI-assisted development on this project.

## Project

AgriSort AI is a **software-only**, AI-powered Streamlit web app that analyses apple images and classifies visible quality as **Good**, **Damaged** or **Rotten**. It's an academic engineering project. Prioritise a small, working, demonstrable prototype over breadth.

## Hard scope rules

- Software only. **No** IoT, Arduino, ESP32, Raspberry Pi, sensors, conveyor belts or physical sorting.
- Stack: Python 3.12, Streamlit (UI), PyTorch + Ultralytics YOLO (ML), OpenCV (image processing), SQLite (database), Plotly (charts).
- Don't add other frameworks or services without asking.

## Honesty rules (critical)

- **Never fabricate** ML predictions, datasets, labels or evaluation metrics.
- **Never claim the model is trained** until training has actually run and produced weights in `models/`.
- UI placeholders must say clearly that the model isn't trained. Charts must only plot real saved records. When there's no data, show an empty-state message, not sample numbers.
- `core/classifier.py::predict()` runs the trained model v1. If the weights are missing or inference fails, it raises `ModelUnavailableError` / `InferenceError`, which the UI shows. It never returns a made-up result. Keep it that way.
- Report metrics only from a real held-out test set, and say which dataset and split were used.

## Environment rules

- Machine: MacBook Air M4, 16 GB RAM, macOS, Apple Silicon (arm64). Use `device="mps"` for PyTorch/YOLO training when available.
- Virtual environment: `.venv` (Python 3.12.16). **Never delete, recreate or modify `.venv`** except by installing packages the user has approved.
- Don't overwrite existing files without reading them first.
- Pin any new dependency in `requirements.txt`.

## Code layout

- `app/main.py`: entry point (`st.set_page_config`, `init_db()`, `st.navigation`, sidebar).
- `app/views/`: one file per page. (It's named `views/`, not `pages/`, to avoid Streamlit's automatic `pages/` handling.)
- `app/components/ui.py`: shared CSS and UI helpers. Escape any user text inserted into HTML.
- `app/core/`: non-UI logic: `config.py` (paths, classes, colours), `database.py`, `image_utils.py`, `classifier.py`.
- Imports are relative to `app/` (for example `from core.config import ...`), because Streamlit adds the script's folder to `sys.path`.
- Datasets go in `data/raw` and `data/processed`, and both are gitignored. Model weights go in `models/`, also gitignored **except** the released `models/agrisort_good_rotten_v1.pt` and `models/README.md`. Training scripts go in `scripts/`, which is tracked.

## Commands

```bash
streamlit run app/main.py        # run the app (from the project root)
python -m pytest tests -q        # run tests (they use a temporary DB via AGRISORT_DB_PATH)
```

## Working style

- The user is a beginner. Explain what you're doing and why in plain language.
- Work incrementally. Verify the app runs (tests plus `streamlit run`) after each change, before moving on.
- Use current Streamlit APIs (`width="stretch"`, not the deprecated `use_container_width`).
- Keep the UI professional and modern, suitable for an academic demo.

## Dataset rules

- Dataset: FruitNet apples (CC BY 4.0), relabelled by people. The source, licence, attribution and audit results are in `docs/DATASET.md`. Keep that file up to date.
- **Never assign Damaged or Rotten labels automatically.** Labels come from people using `labelling/label_tool.py`, which saves to `data/labels/`. Never edit or delete label files without asking.
- Splits must be group-aware (capture bursts plus near-duplicates). `summary.json` must report `groups_split_across_sets == 0`.
- Pipeline: `scripts/fruitnet_extract.py` → `fruitnet_audit.py` → labelling tool → `merge_labels.py` → `build_yolo_dataset.py`.
- Model v1 (`models/agrisort_good_rotten_v1.pt`, tracked in git; local paths were removed from its metadata, and the weights are identical to the original, see `models/README.md`) has been trained: 2-class Good vs Rotten, YOLO11n-cls, MPS, imgsz 224, 10 epochs, batch 16, `cls_pw=1.0`. The real results are in `reports/agrisort_good_rotten_v1/REPORT.md`. Never quote metrics from anywhere else. Never use the test split for tuning. Any new training run needs the user's approval and a new `--name`.

## Public repository rules

- The repository is intended to be public (licence: AGPL-3.0-or-later for code and model, CC BY 4.0 for `data/labels/`; see `NOTICE.md`).
- Never commit absolute local paths, usernames, credentials, `data/agrisort.db` or backups. Before committing new training logs or reports, replace the project folder path with `<project-root>`. Also clean the metadata of any `.pt` file you publish the same way as `models/README.md` describes.
- Keep the FruitNet attribution wherever dataset images appear (reports, screenshots).
