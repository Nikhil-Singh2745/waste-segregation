"""Test the WHOLE pipeline (detector + classifier together) on held-out photos.

For every real object in the test photos we ask: did the system draw a box on it
(overlap IoU >= 0.5) AND give it the right class? That fraction is the end-to-end
success rate. Only objects whose class we know are scored for classification.

Run:  python -m wastesort.evaluate --data data
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from .config import CLASSES, UNKNOWN
from .infer import WasteSorter


def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area = lambda t: (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--out", default="runs/pipeline")
    ap.add_argument("--save", type=int, default=24, help="how many annotated example photos to save")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sorter = WasteSorter()
    n = len(CLASSES)
    found = np.zeros(n + 1, int); total = np.zeros(n + 1, int)       # last slot = class unknown
    cm = np.zeros((n, n), int)
    n_pred = n_pred_ok = 0
    files = sorted(glob.glob(f"{args.data}/det/{args.split}/images/*.jpg"))
    multi_saved = 0
    for f in files:
        img = cv2.imread(f)
        h, w = img.shape[:2]
        rows = np.array([[float(v) for v in l.split()] for l in open(f.replace("/images/", "/labels/")[:-4] + ".txt")]).reshape(-1, 5)
        gt = np.stack([(rows[:, 1] - rows[:, 3] / 2) * w, (rows[:, 2] - rows[:, 4] / 2) * h,
                       (rows[:, 1] + rows[:, 3] / 2) * w, (rows[:, 2] + rows[:, 4] / 2) * h], 1)
        gt_cls = rows[:, 0].astype(int)
        res = sorter.predict(img, args.conf)
        pred = np.array([o["box"] for o in res["objects"]], float).reshape(-1, 4)
        n_pred += len(pred)
        ious = iou_matrix(gt, pred) if len(pred) and len(gt) else np.zeros((len(gt), len(pred)))
        used = set()
        for gi in np.argsort(-ious.max(1) if ious.size else np.zeros(len(gt))):
            slot = n if gt_cls[gi] == UNKNOWN else gt_cls[gi]
            total[slot] += 1
            cand = [(ious[gi, pi], pi) for pi in range(len(pred)) if pi not in used and ious[gi, pi] >= 0.5]
            if cand:
                _, pi = max(cand)
                used.add(pi); found[slot] += 1; n_pred_ok += 1
                if slot < n:
                    cm[slot, res["objects"][pi]["class_id"]] += 1
        if multi_saved < args.save and (len(gt) > 1 or multi_saved < args.save // 2):
            cv2.imwrite(f"{args.out}/example_{multi_saved:02d}.jpg", res["image"]); multi_saved += 1

    known_total, known_found, right = total[:n].sum(), found[:n].sum(), cm.trace()
    lines = [f"photos: {len(files)}   objects: {total.sum()}   boxes drawn: {n_pred}",
             f"detector  recall    {found.sum() / max(1, total.sum()):.3f}   (objects that got a box)",
             f"detector  precision {n_pred_ok / max(1, n_pred):.3f}   (boxes that sit on a real object)",
             f"classifier accuracy on found objects   {right / max(1, known_found):.3f}",
             f"END-TO-END success (found AND right class) {right / max(1, known_total):.3f}", "",
             f"{'class':16s} {'objects':>8s} {'found':>7s} {'correct':>8s} {'success':>8s}"]
    for i, name in enumerate(CLASSES):
        lines.append(f"{name:16s} {total[i]:8d} {found[i]:7d} {cm[i, i]:8d} {cm[i, i] / max(1, total[i]):8.3f}")
    text = "\n".join(lines)
    print(text)
    open(f"{args.out}/report.txt", "w").write(text + "\n")
    json.dump(dict(total=total.tolist(), found=found.tolist(), confusion=cm.tolist(), classes=CLASSES,
                   end_to_end=float(right / max(1, known_total))), open(f"{args.out}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
