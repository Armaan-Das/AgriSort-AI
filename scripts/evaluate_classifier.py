"""Evaluate a trained AgriSort AI classifier on the held-out TEST split (run once per model).

Outputs (reports/<run>/test_evaluation/):
  metrics.json           accuracy, balanced accuracy, per-class P/R/F1, macro F1,
                         confusion matrix, group-bootstrap 95% intervals, inference time
  predictions.csv        one row per test image: true, predicted, confidence, group, correct
  confusion_matrix.png
  misclassified.jpg      contact sheet of every wrong prediction (if any)

Usage:
    python scripts/evaluate_classifier.py --weights models/agrisort_good_rotten_v1.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "fruitnet_cls"
AUDIT_CSV = PROJECT_ROOT / "data" / "interim" / "fruitnet_audit" / "images.csv"


def time_inference(model, paths, device, imgsz, warmup=5):
    for p in paths[:warmup]:
        model.predict(str(p), device=device, imgsz=imgsz, verbose=False)
    ms = []
    for p in paths:
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        model.predict(str(p), device=device, imgsz=imgsz, verbose=False)
        if device == "mps":
            torch.mps.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": round(statistics.mean(ms), 2), "median_ms": round(statistics.median(ms), 2),
            "p95_ms": round(float(np.percentile(ms, 95)), 2), "n_images": len(ms),
            "includes": "image loading, preprocessing, model forward pass and postprocessing (batch size 1)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    run = weights.stem
    out = PROJECT_ROOT / "reports" / run / f"{args.split}_evaluation"
    out.mkdir(parents=True, exist_ok=True)

    manifest = [m for m in csv.DictReader(open(DATA_DIR / "manifest.csv")) if m["split"] == args.split]
    audit = {r["image_id"]: r for r in csv.DictReader(open(AUDIT_CSV))}
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = YOLO(str(weights), task="classify")
    names = model.names  # {0: 'Good', 1: 'Rotten'}
    classes = [names[i] for i in sorted(names)]

    rows = []
    for m in manifest:
        r = model.predict(str(DATA_DIR / m["path"]), device=device, imgsz=args.imgsz, verbose=False)[0]
        probs = r.probs.data.cpu().numpy()
        pred = names[int(probs.argmax())]
        rows.append({"image_id": m["image_id"], "path": m["path"], "true": m["label"], "pred": pred,
                     "confidence": round(float(probs.max()), 4),
                     **{f"p_{c}": round(float(probs[i]), 4) for i, c in enumerate(classes)},
                     "group_id": audit[m["image_id"]]["group_id"], "correct": pred == m["label"]})

    y_true = [r["true"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    p, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Group (cluster) bootstrap: resample independent groups, not images.
    rng = np.random.default_rng(args.seed)
    groups = sorted({r["group_id"] for r in rows})
    by_group = {g: [r for r in rows if r["group_id"] == g] for g in groups}
    boot_acc, boot_bal = [], []
    for _ in range(args.bootstrap):
        sample = [r for g in rng.choice(groups, size=len(groups), replace=True) for r in by_group[g]]
        t = [r["true"] for r in sample]
        if len(set(t)) < len(classes):
            continue
        pr = [r["pred"] for r in sample]
        boot_acc.append(accuracy_score(t, pr))
        boot_bal.append(balanced_accuracy_score(t, pr))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    paths = [DATA_DIR / m["path"] for m in manifest]
    timing = {device: time_inference(model, paths, device, args.imgsz)}
    if device != "cpu":
        timing["cpu"] = time_inference(YOLO(str(weights), task="classify"), paths, "cpu", args.imgsz)

    metrics = {
        "model": str(weights), "split": args.split, "n_images": len(rows),
        "n_independent_groups": len(groups),
        "class_support": {c: int(s) for c, s in zip(classes, sup)},
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "per_class": {c: {"precision": round(float(p[i]), 4), "recall": round(float(rc[i]), 4),
                          "f1": round(float(f1[i]), 4), "support": int(sup[i])} for i, c in enumerate(classes)},
        "macro_f1": round(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0), 4),
        "confusion_matrix": {"labels": classes, "rows_true_cols_pred": cm.tolist()},
        "group_bootstrap_95ci": {"accuracy": ci(boot_acc), "balanced_accuracy": ci(boot_bal),
                                 "resamples_used": len(boot_acc), "unit": "leakage group"},
        "misclassified": sum(not r["correct"] for r in rows),
        "inference_time_per_image": timing,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Greens")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
           xlabel="Predicted", ylabel="True", title=f"{args.split} set (n={len(rows)})")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)

    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        tiles = []
        for r in wrong:
            im = cv2.resize(cv2.imread(str(DATA_DIR / r["path"])), (200, 200))
            cv2.putText(im, f"T:{r['true']} P:{r['pred']} {r['confidence']:.2f}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(im, f"T:{r['true']} P:{r['pred']} {r['confidence']:.2f}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            tiles.append(im)
        while len(tiles) % 6:
            tiles.append(np.zeros((200, 200, 3), np.uint8))
        cv2.imwrite(str(out / "misclassified.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
