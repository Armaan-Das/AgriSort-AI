"""Model Information page - genuine training configuration and evaluation results of model v1.

Every value is read from reports/agrisort_good_rotten_v1/ (written by the training and
evaluation scripts). Nothing is typed in by hand except the limitation descriptions.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import footer, info_card, page_header
from core.classifier import is_model_available, model_status
from core.config import CLASS_COLOURS, MODEL_CLASSES, MODEL_NAME, PLANNED_CLASSES
from core.model_info import LIMITATIONS, load_report

page_header("Model Information", f"Training configuration and held-out evaluation of `{MODEL_NAME}`")

rep = load_report()
if rep["missing"]:
    st.warning(f"Some report files are missing: {', '.join(rep['missing'])}. "
               "Only the available results are shown.", icon=":material/warning:")
train, test, data, ver, epochs = rep["training"], rep["test"], rep["dataset"], rep["verification"], rep["epochs"]

# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", "YOLO11n-cls", border=True, help="1.53 M parameters (2-class head), Ultralytics")
c2.metric("Classes", " / ".join(MODEL_CLASSES), border=True,
          help=f"Planned for the final model: {', '.join(PLANNED_CLASSES)}")
c3.metric("Status", "Ready" if is_model_available() else "Weights missing", border=True, help=model_status())
if test:
    c4.metric("Test balanced accuracy", f"{test['balanced_accuracy']:.1%}", border=True)

st.info("This prototype classifies **Good** vs **Rotten** only. **Damaged** apples and detection of "
        "individual apples in multi-apple photos are planned for future development.",
        icon=":material/info:")

# ---------------------------------------------------------------------------
if test:
    st.markdown("#### Held-out test set results")
    ci = test["group_bootstrap_95ci"]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Accuracy", f"{test['accuracy']:.1%}", border=True,
              help=f"95% interval (group bootstrap): {ci['accuracy'][0]:.1%} – {ci['accuracy'][1]:.1%}")
    t2.metric("Balanced accuracy", f"{test['balanced_accuracy']:.1%}", border=True,
              help=f"95% interval (group bootstrap): {ci['balanced_accuracy'][0]:.1%} – "
                   f"{ci['balanced_accuracy'][1]:.1%}")
    t3.metric("Macro F1", f"{test['macro_f1']:.3f}", border=True)
    t4.metric("Misclassified", f"{test['misclassified']} / {test['n_images']}", border=True)
    st.caption(f"Test set: {test['n_images']} images from {test['n_independent_groups']} independent groups, "
               "evaluated once after training. 95% intervals (hover the metrics) come from resampling "
               f"groups: accuracy {ci['accuracy'][0]:.1%}–{ci['accuracy'][1]:.1%}, balanced accuracy "
               f"{ci['balanced_accuracy'][0]:.1%}–{ci['balanced_accuracy'][1]:.1%}.")

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("##### Per-class metrics")
        per = pd.DataFrame([{"Class": c, "Precision": v["precision"], "Recall": v["recall"],
                             "F1-score": v["f1"], "Test images": v["support"]}
                            for c, v in test["per_class"].items()])
        st.dataframe(per, hide_index=True, width="stretch", column_config={
            "Precision": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "F1-score": st.column_config.NumberColumn(format="%.3f")})
        st.markdown("##### Inference time per image")
        timing = pd.DataFrame([{"Device": d.upper(), "Mean (ms)": t["mean_ms"], "Median (ms)": t["median_ms"],
                                "95th pct (ms)": t["p95_ms"]}
                               for d, t in test["inference_time_per_image"].items()])
        st.dataframe(timing, hide_index=True, width="stretch")
        st.caption("Batch size 1, Apple M4; includes image loading, preprocessing and postprocessing.")
    with right:
        st.markdown("##### Confusion matrix")
        cm = test["confusion_matrix"]
        labels, z = cm["labels"], cm["rows_true_cols_pred"]
        fig = go.Figure(go.Heatmap(
            z=z, x=[f"Pred. {l}" for l in labels], y=[f"True {l}" for l in labels],
            colorscale="Greens", showscale=False, text=z, texttemplate="%{text}",
            textfont={"size": 20}))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
if epochs:
    st.markdown("#### Learning curves")
    ep = [int(r["epoch"]) for r in epochs]
    best = int(train["best_epoch_by_val_accuracy"]["epoch"]) if train else None
    l1, l2 = st.columns(2)
    for col, title, keys in (
        (l1, "Loss", [("train_loss", "Train loss (augmented)"), ("val_loss", "Validation loss")]),
        (l2, "Accuracy", [("train_accuracy", "Train accuracy (un-augmented)"),
                          ("val_accuracy", "Validation accuracy")]),
    ):
        fig = go.Figure()
        for key, name in keys:
            fig.add_trace(go.Scatter(x=ep, y=[r[key] for r in epochs], mode="lines+markers", name=name))
        if best:
            fig.add_vline(x=best, line_dash="dash", line_color="grey",
                          annotation_text=f"best.pt (epoch {best})", annotation_position="bottom right")
        fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10),
                          plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.25),
                          xaxis=dict(title="Epoch", dtick=1))
        col.plotly_chart(fig, width="stretch")
    tr_max = max(r["train_accuracy"] for r in epochs)
    va = [r["val_accuracy"] for r in epochs]
    st.caption(f"Training accuracy reaches {tr_max:.0%} while validation accuracy varies between "
               f"{min(va):.0%} and {max(va):.0%}: the model overfits the small training set, and the small "
               "validation set is noisy. The checkpoint with the best validation accuracy was kept; "
               "the test set was not used for any decision.")

# ---------------------------------------------------------------------------
left, right = st.columns(2, gap="large")
with left:
    st.markdown("#### Training configuration")
    if train:
        s = train["settings"]
        cw = train.get("class_weight_check") or {}
        rows = [
            ("Architecture", "YOLO11n-cls (transfer learning)"),
            ("Pretrained weights", "ImageNet (" + s["pretrained_weights"].split("/")[-1] + ")"),
            ("Image size", f"{s['imgsz']} × {s['imgsz']}"),
            ("Epochs", s["epochs"]),
            ("Batch size", s["batch"]),
            ("Random seed", s["seed"]),
            ("Class-weighted loss", f"cls_pw={s['cls_pw']} → weights "
                                    + ", ".join(f"{n} {w:.3f}" for n, w in
                                                zip(cw.get("class_names", []), cw.get("loss_function_class_weights", [])))
                                    + (" (verified)" if cw.get("applied_correctly") else "")),
            ("Device", train["device_used"].upper() + " (Apple M4)"),
            ("Training time", f"{train['training_duration_seconds']:.0f} s"),
            ("Best epoch (val. accuracy)", f"{train['best_epoch_by_val_accuracy']['epoch']} "
                                           f"({train['best_epoch_by_val_accuracy']['val_accuracy']:.1%})"),
            ("Software", f"PyTorch {train['hardware']['torch']}, Ultralytics {train['hardware']['ultralytics']}"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Setting", "Value"]).astype(str), hide_index=True,
                     width="stretch")
with right:
    st.markdown("#### Dataset distribution")
    if data:
        counts = data["counts"]
        ds = pd.DataFrame([{"Split": s.capitalize(), **{c: counts[s][c] for c in MODEL_CLASSES},
                            "Total": sum(counts[s].values())} for s in ("train", "val", "test")])
        st.dataframe(ds, hide_index=True, width="stretch")
        fig = go.Figure([go.Bar(name=c, x=ds["Split"], y=ds[c], marker_color=CLASS_COLOURS[c])
                         for c in MODEL_CLASSES])
        fig.update_layout(barmode="stack", height=230, margin=dict(l=10, r=10, t=10, b=10),
                          plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    if ver:
        st.caption(f"Group-aware split, leakage check {'passed' if ver['passed'] else 'FAILED'}: "
                   f"{ver['groups_spanning_splits']} groups and {ver['sessions_spanning_splits']} capture "
                   f"sessions shared between splits. Labels: manual relabelling of FruitNet apple images.")

# ---------------------------------------------------------------------------
st.markdown("#### Known limitations")
cols = st.columns(2)
for i, (title, body) in enumerate(LIMITATIONS):
    with cols[i % 2]:
        info_card(title, body)
        st.write("")

st.markdown("#### Data source and licence")
st.caption("Apple images from FruitNet (Meshram & Patil, 2022), Mendeley Data, DOI 10.17632/b6fftwbr2v.3, "
           "licensed under CC BY 4.0. Images were relabelled by the AgriSort AI team into Good / Damaged / "
           "Rotten (original labels: Good / Bad / Mixed). Full report: "
           f"`reports/{MODEL_NAME}/REPORT.md`.")

footer()
