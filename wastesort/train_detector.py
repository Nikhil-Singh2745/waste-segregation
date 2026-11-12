"""Train the detector CNN from scratch.

Run:  python -m wastesort.train_detector --data data --epochs 80
Writes weights/detector.pt and training curves in runs/detector/.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, generalized_box_iou_loss

from .config import DET_SIZE, DET_STRIDE
from .data import DetDataset, det_collate
from .models import WasteDetector, count_params, decode_boxes, detections


def detector_loss(out, heat, pos):
    """Three parts: is-there-an-object heat-map, box overlap, centre offset."""
    # 1) heat-map: "focal loss" - concentrates on the cells the network still gets wrong
    p = torch.sigmoid(out[:, 0].float()).clamp(1e-4, 1 - 1e-4)
    is_pos = heat.eq(1)
    pos_loss = -(torch.log(p) * (1 - p) ** 2)[is_pos].sum()
    neg_loss = -(torch.log(1 - p) * p ** 2 * (1 - heat) ** 4)[~is_pos].sum()
    n = max(1, len(pos))
    hm = (pos_loss + neg_loss) / n
    if not len(pos):
        return hm, hm.detach(), torch.zeros(()), torch.zeros(())
    # 2) boxes, only at the cells that really contain an object centre
    b, iy, ix = pos[:, 0].long(), pos[:, 1].long(), pos[:, 2].long()
    pred = decode_boxes(out.float())[b, iy, ix]
    giou = generalized_box_iou_loss(pred, pos[:, 3:7], reduction="mean")
    # 3) centre offset inside the cell
    s = float(DET_STRIDE)
    tx = (pos[:, 3] + pos[:, 5]) / 2 / s - ix
    ty = (pos[:, 4] + pos[:, 6]) / 2 / s - iy
    off = F.l1_loss(torch.sigmoid(out[b, 1:3, iy, ix].float()), torch.stack([tx, ty], 1).clamp(0, 1))
    return hm + 2.0 * giou + off, hm.detach(), giou.detach(), off.detach()


def average_precision(preds, gts, iou_thr=0.5):
    """AP@0.5 over a dataset (class-agnostic) + best-F1 operating point."""
    scores, hits, total = [], [], 0
    for det, gt in zip(preds, gts):
        total += len(gt)
        used = torch.zeros(len(gt), dtype=torch.bool)
        order = det[:, 4].argsort(descending=True)
        ious = box_iou(det[order, :4], gt) if len(gt) and len(det) else torch.zeros(len(det), len(gt))
        for k in range(len(order)):
            hit = False
            if len(gt):
                iou, j = (ious[k] * (~used)).max(0)
                if iou >= iou_thr:
                    used[j] = True
                    hit = True
            scores.append(det[order[k], 4].item()); hits.append(hit)
    if not scores or not total:
        return dict(ap50=0.0, precision=0.0, recall=0.0, f1=0.0, conf=0.3)
    o = np.argsort(scores)[::-1]
    hits, scores = np.array(hits)[o], np.array(scores)[o]
    tp = np.cumsum(hits); fp = np.cumsum(~hits)
    rec, prec = tp / total, tp / (tp + fp)
    mono = np.maximum.accumulate(prec[::-1])[::-1]
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * mono))
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    k = int(f1.argmax())
    return dict(ap50=ap, precision=float(prec[k]), recall=float(rec[k]), f1=float(f1[k]), conf=float(scores[k]))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for imgs, _, _, boxes in loader:
        out = model(imgs.to(device))
        preds += detections(out, conf=0.05)
        gts += boxes
    return average_precision(preds, gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--size", type=int, default=DET_SIZE)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count()))
    ap.add_argument("--limit", type=int, default=None, help="use only N training images (quick tests)")
    ap.add_argument("--weights", default="weights/detector.pt")
    ap.add_argument("--out", default="runs/detector")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.weights) or ".", exist_ok=True)
    torch.manual_seed(0)

    train_ds = DetDataset(args.data, "train", train=True, size=args.size, limit=args.limit)
    val_ds = DetDataset(args.data, "val", size=args.size)
    test_ds = DetDataset(args.data, "test", size=args.size)
    kw = dict(num_workers=args.workers, collate_fn=det_collate, pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, 64, **kw)
    test_dl = DataLoader(test_ds, 64, **kw)

    model = WasteDetector().to(device).to(memory_format=torch.channels_last)
    print(f"device={device}  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}  weights={count_params(model) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, agg, steps = time.time(), np.zeros(4), 0
        for imgs, heat, pos, _ in train_dl:
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            heat, pos = heat.to(device), pos.to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                out = model(imgs)
            loss, hm, giou, off = detector_loss(out, heat, pos)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            agg += [loss.item(), hm.item(), giou.item(), off.item()]; steps += 1
        agg /= steps
        m = evaluate(model, val_dl, device)
        history.append(dict(epoch=epoch, loss=agg[0], heat=agg[1], giou=agg[2], off=agg[3], **m))
        flag = ""
        if m["ap50"] >= best:
            best, flag = m["ap50"], "  <- best, saved"
            torch.save({"model": model.state_dict(), "size": args.size, "conf": m["conf"], "val": m}, args.weights)
        print(f"epoch {epoch:3d}/{args.epochs}  loss {agg[0]:.3f} (heat {agg[1]:.3f} box {agg[2]:.3f})  "
              f"val AP50 {m['ap50']:.3f}  P {m['precision']:.3f} R {m['recall']:.3f}  ({time.time() - t0:.0f}s){flag}", flush=True)

    ck = torch.load(args.weights, map_location=device)
    model.load_state_dict(ck["model"])
    m = evaluate(model, test_dl, device)
    print(f"\nTEST  AP50 {m['ap50']:.3f}   precision {m['precision']:.3f}  recall {m['recall']:.3f}  (at confidence {m['conf']:.2f})")
    json.dump(dict(history=history, test=m), open(f"{args.out}/results.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = [h["epoch"] for h in history]
    ax[0].plot(ep, [h["loss"] for h in history]); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    for k in ("ap50", "precision", "recall"):
        ax[1].plot(ep, [h[k] for h in history], label=k)
    ax[1].set_title("Validation"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.out}/curves.png", dpi=130)


if __name__ == "__main__":
    main()
