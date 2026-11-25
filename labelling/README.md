# AgriSort AI labelling tool

This is a standalone Streamlit app for checking FruitNet apple images by hand. It's separate from the AgriSort AI website.

## Launch

```bash
cd ~/Developer/AgriSort-AI
source .venv/bin/activate
streamlit run labelling/label_tool.py --server.port 8502
```

Open http://localhost:8502. The website keeps using port 8501, so both can run at the same time.

## Controls

| Key | Action |
|---|---|
| `1` | Good |
| `2` | Damaged |
| `3` | Rotten |
| `4` | Exclude |
| `←` / `→` | Previous / next image |
| `N` | Jump to the next unlabelled image |

Click once on the page before using the keys, so the browser window has focus. Streamlit reserves `R` and `C`, so the class shortcuts are number keys.

## Labelling guidelines

- **Good:** no significant visible damage or decay.
- **Damaged:** visible bruises, cuts, cracks or physical damage *without* clear signs of decay.
- **Rotten:** visible signs of decay, mould or decomposition.
- **Exclude:** ambiguous images, several apples in different conditions, or anything you can't classify reliably. *When in doubt, choose Exclude.*

## Workflow

1. Enter your **name or initials** in the sidebar. The tool won't save labels without one.
2. Choose which images to label:
   - **'Bad' folder:** sort these into Damaged, Rotten or Exclude. This is the main job.
   - **'Good' folder:** confirm Good, or relabel or exclude if you spot a defect.
   - **'Mixed' folder:** optional. These are mostly several fruits per image, so most will be Exclude.
3. Choose a batch:
   - **All images** if one person is labelling everything.
   - **Batch 1–6** to split the work between up to six people. The batches don't overlap, and each saves to its own file.
4. Label. Each click saves straight to `data/labels/` and moves to the next unlabelled image.
5. Close the tab whenever you like. Next time, the tool resumes at the first unlabelled image.

Images appear in a fixed random order, so stopping partway still leaves a fair sample. Watch the sidebar counts and aim for similar numbers of Good, Damaged and Rotten images.

## Afterwards

```bash
python scripts/merge_labels.py                 # combine batch files and report conflicts
python scripts/build_yolo_dataset.py --balance # build the train/val/test folders
```

## Where labels are saved

| Batch | File |
|---|---|
| All images | `data/labels/fruitnet_labels_all.csv` |
| Batch *k* | `data/labels/fruitnet_labels_batch<k>.csv` |

Each row records the image, label, original FruitNet folder, annotator and time. Use only one browser tab per label file at a time.
