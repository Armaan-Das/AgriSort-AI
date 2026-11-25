"""Train the AgriSort AI Good-vs-Rotten classifier (YOLO11n-cls, transfer learning).

Uses ONLY the train and val splits of data/processed/fruitnet_cls/.
The test split is never touched here (see scripts/evaluate_classifier.py).

What it records (reports/<run>/):
  training_summary.json  settings, device, duration, class-weight verification, best epoch
  epoch_metrics.csv      per epoch: train loss, train accuracy (un-augmented train images),
                         val loss, val accuracy, learning rate
  learning_curves.png    loss and accuracy curves
  ultralytics/           Ultralytics' own results.csv / plots for the run

Weights:
  runs/classify/<run>/weights/{best,last}.pt   (best = highest val top-1 accuracy)
  models/<run>.pt                               copy of best.pt

Usage:
    python scripts/train_classifier.py              # approved config: 10 epochs, 224 px, batch 16
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "fruitnet_cls"
PRETRAINED = PROJECT_ROOT / "models" / "pretrained" / "yolo11n-cls.pt"
RUNS_DIR = PROJECT_ROOT / "runs" / "classify"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def train_accuracy(trainer, loader) -> float:
    """Top-1 accuracy of the EMA model (the one that is validated) on un-augmented train images."""
    model = trainer.ema.ema if trainer.ema else trainer.model
    was_training = model.training
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(trainer.device, non_blocking=True).float()
            cls = batch["cls"].to(trainer.device)
            out = model(img)
            out = out[0] if isinstance(out, (list, tuple)) else out
            correct += (out.argmax(1) == cls).sum().item()
            total += cls.numel()
    model.train(was_training)
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="agrisort_good_rotten_v1")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cls-pw", type=float, default=1.0, help="class-weight power (1.0 = inverse frequency)")
    ap.add_argument("--device", default="auto", help="auto (MPS if available, else CPU), mps or cpu")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.utils import __version__ as ul_version

    if not PRETRAINED.exists():
        raise SystemExit(f"Missing pretrained weights: {PRETRAINED}")
    if (RUNS_DIR / args.name).exists():
        raise SystemExit(f"Run '{args.name}' already exists - choose a new --name (results are never overwritten).")

    report_dir = REPORTS_DIR / args.name
    report_dir.mkdir(parents=True, exist_ok=True)
    epoch_rows: list[dict] = []
    checks: dict = {}
    state: dict = {}

    def on_train_start(trainer):
        # un-augmented copy of the train split, using the same preprocessing as validation
        ds = ClassificationDataset(root=str(DATA_DIR / "train"), args=trainer.args, augment=False,
                                   prefix="train-eval")
        state["train_eval_loader"] = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False,
                                                                 num_workers=0)

    def on_train_epoch_end(trainer):
        if "class_weights" in checks:
            return
        model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        counts = trainer.get_class_counts()
        expected = (1.0 / counts) ** args.cls_pw
        expected = expected / expected.mean()
        model_w = getattr(model, "class_weights", None)
        crit = getattr(model, "criterion", None)
        crit_w = getattr(crit, "class_weights", None) if crit is not None else None
        checks["class_weights"] = {
            "class_names": [trainer.data["names"][i] for i in range(trainer.data["nc"])],
            "train_image_counts": counts.tolist(),
            "expected_weights": np.round(expected, 4).tolist(),
            "model_class_weights": None if model_w is None else np.round(model_w.cpu().numpy(), 4).tolist(),
            "loss_function_class_weights": None if crit_w is None else np.round(crit_w.cpu().numpy(), 4).tolist(),
            "loss_function": type(crit).__name__ if crit is not None else None,
        }
        ok = crit_w is not None and np.allclose(crit_w.cpu().numpy().ravel(), expected, atol=1e-3)
        checks["class_weights"]["applied_correctly"] = bool(ok)

    def on_fit_epoch_end(trainer):
        # Ultralytics fires this once more after training when it re-validates best.pt; skip that call.
        if len(epoch_rows) >= trainer.epochs:
            state["final_best_val_accuracy"] = float(trainer.metrics.get("metrics/accuracy_top1", float("nan")))
            return
        m = trainer.metrics
        tl = trainer.tloss  # dict of running-mean loss items in this Ultralytics version
        train_loss = sum(float(v) for v in tl.values()) if isinstance(tl, dict) else float(tl)
        val_loss = sum(float(v) for k, v in m.items() if k.startswith("val/") and "loss" in k)
        if not any(k.startswith("val/") and "loss" in k for k in m):
            val_loss = float("nan")
        epoch_rows.append({
            "epoch": trainer.epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_accuracy": round(train_accuracy(trainer, state["train_eval_loader"]), 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(float(m.get("metrics/accuracy_top1", float("nan"))), 5),
            "lr": round(float(trainer.optimizer.param_groups[0]["lr"]), 7),
        })
        r = epoch_rows[-1]
        print(f"[AgriSort] epoch {r['epoch']}: train_loss={r['train_loss']} train_acc={r['train_accuracy']} "
              f"val_loss={r['val_loss']} val_acc={r['val_accuracy']}")

    device = pick_device(args.device)
    fallback_reason = None
    train_kwargs = dict(data=str(DATA_DIR), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                        seed=args.seed, deterministic=True, cls_pw=args.cls_pw, pretrained=True,
                        project=str(RUNS_DIR), name=args.name, exist_ok=False, plots=True,
                        workers=args.workers, verbose=True)

    def run(dev: str):
        model = YOLO(str(PRETRAINED))
        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        t0 = time.perf_counter()
        model.train(device=dev, **train_kwargs)
        return model, time.perf_counter() - t0

    started = datetime.now()
    try:
        model, seconds = run(device)
    except (RuntimeError, NotImplementedError) as exc:
        if device != "mps":
            raise
        fallback_reason = f"{type(exc).__name__}: {exc}"
        print(f"[AgriSort] MPS failed ({fallback_reason}); falling back to CPU.")
        shutil.rmtree(RUNS_DIR / args.name, ignore_errors=True)
        epoch_rows.clear()
        checks.clear()
        device = "cpu"
        model, seconds = run(device)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    final_model = MODELS_DIR / f"{args.name}.pt"
    shutil.copy2(best, final_model)

    with open(report_dir / "epoch_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        w.writeheader()
        w.writerows(epoch_rows)

    # learning curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in epoch_rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [r["train_loss"] for r in epoch_rows], "o-", label="train loss (augmented)")
    a1.plot(ep, [r["val_loss"] for r in epoch_rows], "s-", label="val loss")
    a1.set(xlabel="epoch", ylabel="loss", title="Loss")
    a2.plot(ep, [r["train_accuracy"] for r in epoch_rows], "o-", label="train accuracy (un-augmented)")
    a2.plot(ep, [r["val_accuracy"] for r in epoch_rows], "s-", label="val accuracy")
    a2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy", ylim=(0, 1.02))
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
        a.set_xticks(ep)
    fig.suptitle(f"{args.name} - YOLO11n-cls, Good vs Rotten (train {sum(1 for _ in (DATA_DIR / 'train').rglob('*.*'))} imgs)")
    fig.tight_layout()
    fig.savefig(report_dir / "learning_curves.png", dpi=150)

    # copy Ultralytics' own outputs (small files only)
    ul_dir = report_dir / "ultralytics"
    ul_dir.mkdir(exist_ok=True)
    for p in save_dir.iterdir():
        if p.is_file() and p.suffix in {".csv", ".png", ".jpg", ".yaml"}:
            shutil.copy2(p, ul_dir / p.name)

    best_row = max(epoch_rows, key=lambda r: r["val_accuracy"])
    summary = {
        "run": args.name,
        "started": started.isoformat(timespec="seconds"),
        "training_duration_seconds": round(seconds, 1),
        "device_used": device,
        "mps_fallback_reason": fallback_reason,
        "hardware": {"machine": platform.machine(), "platform": platform.platform(),
                     "torch": torch.__version__, "ultralytics": ul_version, "python": platform.python_version()},
        "settings": {**{k: v for k, v in train_kwargs.items() if k not in ("project",)},
                     "pretrained_weights": str(PRETRAINED.relative_to(PROJECT_ROOT))},
        "class_weight_check": checks.get("class_weights"),
        "best_epoch_by_val_accuracy": best_row,
        "best_pt_revalidated_val_accuracy": state.get("final_best_val_accuracy"),
        "note_best_checkpoint": "Ultralytics saves best.pt at the epoch with the highest validation "
                                "fitness (top-1 accuracy for classification).",
        "weights": {"best": str(final_model.relative_to(PROJECT_ROOT)),
                    "run_dir": str(save_dir.relative_to(PROJECT_ROOT))},
        "test_set_used": False,
    }
    (report_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
