# Contributing to AgriSort AI

Thank you for helping. AgriSort AI is an academic prototype, so small, careful, well-tested changes are the most useful.

## 1. Set up

Follow the installation steps in the [README](README.md#installation), then install the test tools:

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q      # every test should pass before you change anything
```

## 2. Workflow

1. Update your copy: `git checkout main && git pull`
2. Create a branch for your change: `git checkout -b short-description` (for example `fix-history-export`)
3. Make the change and run the tests: `python -m pytest tests -q`
4. Run the app and check the pages you touched: `streamlit run app/main.py`
5. Commit with a clear message that says *what* changed and *why*: `git commit -m "Add sharpness warning threshold to config"`
6. Push the branch (`git push -u origin short-description`) and open a pull request on GitHub. Describe what you changed and how you tested it.

**Don't push directly to `main`**, and never use `git push --force` on shared branches.

## 3. Project rules

These rules keep the project honest and reproducible. Please read them before contributing.

- **Never fabricate results.** No invented predictions, labels, metrics or sample data. Charts only show real saved records, and an empty database shows an empty-state message.
- **Metrics come from the held-out test set only**, and must say which model, dataset and split they describe. Never tune anything on the test split.
- **Damaged and Rotten labels are assigned by people**, using `labelling/label_tool.py`, never automatically.
- **Dataset splits must be group-aware**: `summary.json` must report `groups_split_across_sets == 0`.
- **Any new training run** needs a new `--name` and must not overwrite `agrisort_good_rotten_v1`. Agree it with the team first.
- **Software only.** No hardware, IoT or physical sorting.
- **Dependencies:** pin new packages in `requirements.txt` and explain in the pull request why they're needed.
- **Code layout:** pages go in `app/views/`, shared UI helpers in `app/components/ui.py`, and non-UI logic in `app/core/`. Imports are relative to `app/` (`from core.config import ...`). Use current Streamlit APIs (`width="stretch"`, not `use_container_width`).

## 4. Never commit

- The FruitNet dataset or anything in `data/raw/`, `data/processed/`, `data/downloads/` or `data/interim/`
- Your local database `data/agrisort.db`, or anything in `data/backups/`
- Credentials: `.env`, `.streamlit/secrets.toml`, API keys or tokens
- **Absolute paths from your computer** (such as `/Users/<you>/...` or `C:\Users\<you>\...`). Training logs from Ultralytics contain them. Replace the project folder with `<project-root>` before committing new logs or reports, and clean the metadata of any `.pt` file (see `models/README.md`).
- New model weights, unless the team has agreed to publish them

`.gitignore` already blocks most of these. Check `git status` before every commit anyway.

## 5. Licence of contributions

By contributing, you agree that your contribution is licensed under the project's licences (see [`NOTICE.md`](NOTICE.md)):
- **AGPL-3.0-or-later** for code and models
- **CC BY 4.0** for annotations in `data/labels/`

## 6. Reporting problems

Open a GitHub issue with:
- what you did
- what you expected
- what happened instead
- your operating system and Python version (`python --version`)

**Don't attach private photos or personal data.**
