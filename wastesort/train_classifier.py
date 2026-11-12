"""Train the classifier CNN from scratch.

Run:  python -m wastesort.train_classifier --data data --epochs 60
Writes weights/classifier.pt plus training curves and a confusion matrix in runs/classifier/.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import CLASSES, CLS_SIZE
from .data import ClsDataset
from .models import WasteClassifier, count_params


@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    """Returns accuracy and the confusion matrix (rows = truth, columns = prediction)."""
    model.eval()
    n = len(CLASSES)
    cm = np.zeros((n, n), int)
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        if tta:   # also look at the picture turned 90/180/270 degrees and average the answers
            for k in (1, 2, 3):
                logits = logits + model(torch.rot90(x, k, dims=(2, 3)))
        for t, p in zip(y.tolist(), logits.argmax(1).tolist()):
            cm[t, p] += 1
    return cm.trace() / max(1, cm.sum()), cm


def report(cm):
    lines = [f"{'class':16s} {'precision':>9s} {'recall':>7s} {'images':>7s}"]
    for i, name in enumerate(CLASSES):
        p = cm[i, i] / max(1, cm[:, i].sum())
        r = cm[i, i] / max(1, cm[i].sum())
        lines.append(f"{name:16s} {p:9.3f} {r:7.3f} {cm[i].sum():7d}")
    return "\n".join(lines)


def save_plots(history, cm, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(history) + 1)
    ax[0].plot(ep, [h["loss"] for h in history]); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="validation")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}/curves.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right"); ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (test set)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(f"{out}/confusion_matrix.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--size", type=int, default=CLS_SIZE)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count()))
    ap.add_argument("--limit", type=int, default=None, help="use only N training images (quick tests)")
    ap.add_argument("--weights", default="weights/classifier.pt")
    ap.add_argument("--out", default="runs/classifier")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.weights) or ".", exist_ok=True)
    torch.manual_seed(0)

    train_ds = ClsDataset(args.data, "train", train=True, size=args.size, limit=args.limit)
    val_ds = ClsDataset(args.data, "val", size=args.size)
    test_ds = ClsDataset(args.data, "test", size=args.size)
    kw = dict(num_workers=args.workers, pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, 256, **kw)
    test_dl = DataLoader(test_ds, 256, **kw)

    model = WasteClassifier().to(device).to(memory_format=torch.channels_last)
    print(f"device={device}  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  weights={count_params(model) / 1e6:.2f}M")

    # classes with fewer pictures get a bigger say in the loss
    counts = train_ds.class_counts()
    class_w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    history, best = [], 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device, enabled=device == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); seen += len(y)
            correct += (logits.argmax(1) == y).sum().item()
        val_acc, _ = evaluate(model, val_dl, device)
        history.append(dict(epoch=epoch, loss=tot / seen, train_acc=correct / seen, val_acc=float(val_acc)))
        flag = ""
        if val_acc >= best:
            best, flag = val_acc, "  <- best, saved"
            torch.save({"model": model.state_dict(), "classes": CLASSES, "size": args.size, "val_acc": float(val_acc)}, args.weights)
        print(f"epoch {epoch:3d}/{args.epochs}  loss {tot / seen:.3f}  train {correct / seen:.3f}  val {val_acc:.3f}  "
              f"({time.time() - t0:.0f}s){flag}", flush=True)

    # final exam on pictures the model never saw, using the best checkpoint
    model.load_state_dict(torch.load(args.weights, map_location=device)["model"])
    acc, cm = evaluate(model, test_dl, device)
    acc_tta, cm_tta = evaluate(model, test_dl, device, tta=True)
    print(f"\nTEST accuracy: {acc:.3f}   (with 4-rotation averaging: {acc_tta:.3f})\n{report(cm_tta)}")
    json.dump(dict(history=history, test_acc=float(acc), test_acc_tta=float(acc_tta), confusion=cm_tta.tolist(), classes=CLASSES),
              open(f"{args.out}/results.json", "w"), indent=1)
    save_plots(history, cm_tta, args.out)


if __name__ == "__main__":
    main()
