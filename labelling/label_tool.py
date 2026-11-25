"""AgriSort AI - standalone image labelling tool (separate from the website).

Launch from the project root:
    streamlit run labelling/label_tool.py --server.port 8502

Every click is saved immediately to data/labels/*.csv, so progress is never lost.
Keyboard: 1 Good · 2 Damaged · 3 Rotten · 4 Exclude · ← Previous · → Next · N next unlabelled
"""

from __future__ import annotations

import collections
import sys
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dataset_common import (  # noqa: E402
    EXCLUDE,
    IMAGES_CSV,
    LABEL_FIELDS,
    LABELS_DIR,
    MERGED_LABELS_CSV,
    NUM_BATCHES,
    RAW_DIR,
    TARGET_CLASSES,
    label_file_for,
    read_csv,
    write_csv_atomic,
)

st.set_page_config(page_title="AgriSort AI · Labelling Tool", page_icon=":material/label:",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; max-width: 1400px; }
      .lt-badge { display:inline-block; padding:.2rem .7rem; border-radius:999px;
                  font-weight:600; color:#fff; }
      .lt-meta { color:#5B6B5E; font-size:.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

LABEL_COLOURS = {"Good": "#2E7D32", "Damaged": "#F9A825", "Rotten": "#C62828", EXCLUDE: "#607D8B"}
SHORTCUTS = {"Good": "1", "Damaged": "2", "Rotten": "3", EXCLUDE: "4"}
QUEUES = {
    "Bad": "FruitNet 'Bad' folder → Damaged / Rotten",
    "Good": "FruitNet 'Good' folder → verify Good",
    "Mixed": "FruitNet 'Mixed' folder (mostly Exclude)",
}
# Review mode: re-check a list of flagged images. Every decision is logged with the
# original label, and the reviewed label becomes the label used for training.
REVIEW_LIST = LABELS_DIR / "review_round1_flagged.csv"
REVIEW_LOG = LABELS_DIR / "review_round1_decisions.csv"
REVIEW_FIELDS = ["image_id", "original_label", "reviewed_label", "decision", "annotator", "reviewed_at"]
if REVIEW_LIST.exists():
    QUEUES = {"Review": "Review flagged images (round 1)", **QUEUES}
GUIDELINES = """
**Good** – no significant visible damage or decay.

**Damaged** – visible bruises, cuts, cracks or physical damage **without** clear signs of decay.

**Rotten** – visible signs of decay, mould or decomposition.

**Exclude** – ambiguous; several apples with different conditions; not an apple;
or cannot be classified reliably. *When in doubt, Exclude.*
"""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_images() -> list[dict]:
    rows = read_csv(IMAGES_CSV)
    rows = [r for r in rows if r["readable"] == "True" and not r["exact_dup_of"]]
    return sorted(rows, key=lambda r: int(r["queue_order"]))


@st.cache_data(max_entries=64)
def load_display_image(image_id: str, size: int = 560):
    img = cv2.imread(str(RAW_DIR / image_id), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_labels(path: Path) -> dict[str, dict]:
    return {r["image_id"]: r for r in read_csv(path)}


def combined_label_counts() -> collections.Counter:
    """Counts across every label file (latest label per image wins)."""
    latest: dict[str, dict] = {}
    for f in sorted(LABELS_DIR.glob("fruitnet_labels_*.csv")):
        if f == MERGED_LABELS_CSV:
            continue
        for r in read_csv(f):
            if r["image_id"] not in latest or r["labelled_at"] > latest[r["image_id"]]["labelled_at"]:
                latest[r["image_id"]] = r
    return collections.Counter(r["label"] for r in latest.values())


if not IMAGES_CSV.exists():
    st.error("Audit file not found. Run `python scripts/fruitnet_audit.py` first.")
    st.stop()

images = load_images()

# ---------------------------------------------------------------------------
# Sidebar - session settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### AgriSort AI · Labelling")
    annotator = st.text_input("Annotator name / initials", key="annotator",
                              placeholder="e.g. AD").strip()
    queue = st.radio("Images to label", list(QUEUES), format_func=QUEUES.get, key="queue")
    review_mode = queue == "Review"
    if review_mode:
        batch = "all"
        st.caption("Review mode: **1–4** sets the label · **Enter** confirms the current label. "
                   f"Decisions are logged in `{REVIEW_LOG.name}` with the original label.")
    else:
        batch = st.selectbox(
            "Batch", ["all"] + [str(b) for b in range(1, NUM_BATCHES + 1)],
            format_func=lambda b: "All images (one person)" if b == "all" else f"Batch {b} of {NUM_BATCHES}",
            key="batch",
            help="Batches are fixed and non-overlapping, so up to six people can label in parallel. "
                 "Each batch saves to its own CSV; merge them with scripts/merge_labels.py.",
        )
    label_path = label_file_for(batch)
    st.caption(f"Saving to `{label_path.relative_to(LABELS_DIR.parents[1])}`")

    st.divider()
    st.markdown("**Labelled so far (all files)**")
    counts = combined_label_counts()
    c1, c2 = st.columns(2)
    for col, name in zip([c1, c2, c1, c2], TARGET_CLASSES + [EXCLUDE]):
        col.metric(name, counts.get(name, 0))
    st.caption("Aim for similar Good, Damaged and Rotten counts.")

    with st.expander("Labelling guidelines", expanded=True):
        st.markdown(GUIDELINES)
    st.caption("Keys: **1** Good · **2** Damaged · **3** Rotten · **4** Exclude · "
               "**←/→** navigate · **N** next unlabelled")

if review_mode:
    flagged = {r["image_id"]: r for r in read_csv(REVIEW_LIST)}
    items = [r for r in images if r["image_id"] in flagged]
else:
    flagged = {}
    items = [r for r in images if r["source_quality"] == queue and (batch == "all" or r["batch"] == batch)]
if not items:
    st.info("No images in this selection.")
    st.stop()

labels = load_labels(label_path)
reviews = {r["image_id"]: r for r in read_csv(REVIEW_LOG)} if review_mode else {}
pos_key = f"pos::{queue}::{batch}"


def is_done(image_id: str) -> bool:
    return image_id in (reviews if review_mode else labels)


def first_unlabelled(start: int = 0) -> int | None:
    n = len(items)
    for k in range(n):
        i = (start + k) % n
        if not is_done(items[i]["image_id"]):
            return i
    return None


if pos_key not in st.session_state:  # resume where the last session stopped
    st.session_state[pos_key] = first_unlabelled() or 0


# ---------------------------------------------------------------------------
# Actions (run as callbacks, before the page re-renders)
# ---------------------------------------------------------------------------
def save_label(label: str) -> None:
    if not st.session_state.get("annotator", "").strip():
        st.session_state["flash"] = ("warning", "Enter your name / initials in the sidebar first.")
        return
    item = items[st.session_state[pos_key]]
    now = datetime.now().isoformat(timespec="seconds")
    who = st.session_state["annotator"].strip()
    if review_mode:  # log the decision first, keeping the original label
        log = {r["image_id"]: r for r in read_csv(REVIEW_LOG)}
        original = flagged[item["image_id"]]["label"]
        log[item["image_id"]] = {
            "image_id": item["image_id"], "original_label": original, "reviewed_label": label,
            "decision": "confirmed" if label == original else "changed",
            "annotator": who, "reviewed_at": now,
        }
        write_csv_atomic(REVIEW_LOG, list(log.values()), REVIEW_FIELDS)
        reviews.update(log)
    current = load_labels(label_path)  # re-read from disk so nothing is overwritten
    current[item["image_id"]] = {
        "image_id": item["image_id"], "label": label, "source_quality": item["source_quality"],
        "annotator": who, "labelled_at": now,
    }
    write_csv_atomic(label_path, list(current.values()), LABEL_FIELDS)
    labels.update(current)
    nxt = first_unlabelled(st.session_state[pos_key] + 1)
    if nxt is not None:
        st.session_state[pos_key] = nxt
    st.session_state["flash"] = ("success", f"Saved {item['image_id']} → {label}")


def move(delta: int) -> None:
    st.session_state[pos_key] = (st.session_state[pos_key] + delta) % len(items)


def jump_unlabelled() -> None:
    nxt = first_unlabelled(st.session_state[pos_key] + 1)
    if nxt is not None:
        st.session_state[pos_key] = nxt


# ---------------------------------------------------------------------------
# Main view
# ---------------------------------------------------------------------------
pos = st.session_state[pos_key]
item = items[pos]
done = sum(is_done(r["image_id"]) for r in items)
remaining = len(items) - done
verb = "reviewed" if review_mode else "labelled"

st.markdown(f"#### {QUEUES[queue]} · {'all images' if batch == 'all' else f'batch {batch}'}")
st.progress(done / len(items), text=f"**{done} / {len(items)} {verb}** · {remaining} remaining")

if flash := st.session_state.pop("flash", None):
    kind, msg = flash
    (st.warning if kind == "warning" else st.toast)(msg)

col_img, col_ctl = st.columns([3, 2], gap="large")

with col_img:
    st.image(load_display_image(item["image_id"]), width="stretch")

with col_ctl:
    st.markdown(f"**Image {pos + 1} of {len(items)}**")
    st.markdown(f'<div class="lt-meta">{item["image_id"]} · {item["width"]}×{item["height"]} px · '
                f'FruitNet folder: <b>{item["source_quality"]}</b></div>', unsafe_allow_html=True)
    existing = labels.get(item["image_id"])
    if existing:
        colour = LABEL_COLOURS.get(existing["label"], "#333")
        st.markdown(f'Current label: <span class="lt-badge" style="background:{colour}">'
                    f'{existing["label"]}</span> <span class="lt-meta">by {existing["annotator"]}</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="lt-meta">Not labelled yet</span>', unsafe_allow_html=True)

    if review_mode:
        original = flagged[item["image_id"]]["label"]
        rev = reviews.get(item["image_id"])
        status = (f"✅ Reviewed: **{rev['decision']}** ({rev['original_label']} → {rev['reviewed_label']})"
                  if rev else "⏳ Not reviewed yet")
        st.info(f"Original label: **{original}**. Flagged because no clear defect was visible "
                f"in a thumbnail check. Please confirm or correct it.\n\n{status}",
                icon=":material/flag:")

    if item["cross_folder_conflict"] == "True":
        st.warning("A near-identical image exists in a different FruitNet folder. Check carefully.",
                   icon=":material/compare:")
    elif int(item["group_size"]) > 1:
        st.caption(f"Part of a near-duplicate group of {item['group_size']} images "
                   "(they will be kept in the same split).")

    st.write("")
    for name in TARGET_CLASSES + [EXCLUDE]:
        st.button(f"{name}", key=f"btn_{name}", on_click=save_label, args=(name,),
                  shortcut=SHORTCUTS[name], width="stretch",
                  type="primary" if existing and existing["label"] == name else "secondary")

    if review_mode and existing:
        st.button(f"Confirm current label ({existing['label']})", key="btn_confirm",
                  icon=":material/check:", on_click=save_label, args=(existing["label"],),
                  shortcut="Enter", width="stretch")

    st.write("")
    n1, n2 = st.columns(2)
    n1.button("Previous", icon=":material/arrow_back:", on_click=move, args=(-1,),
              shortcut="Left", width="stretch")
    n2.button("Next", icon=":material/arrow_forward:", on_click=move, args=(1,),
              shortcut="Right", width="stretch")
    st.button("Next unlabelled", icon=":material/skip_next:", on_click=jump_unlabelled,
              shortcut="N", width="stretch", disabled=remaining == 0)

    if remaining == 0:
        st.success(f"Every image in this selection is {verb}.", icon=":material/task_alt:")
