"""The full pipeline: photo -> boxes (detector) -> class of each box (classifier).

Use from Python:
    from wastesort.infer import WasteSorter
    sorter = WasteSorter()                    # loads weights/detector.pt + weights/classifier.pt
    result = sorter.predict(cv2.imread("pile.jpg"))
    cv2.imwrite("out.jpg", result["image"])
"""
import os

import cv2
import numpy as np
import torch

from .config import CLASSES, COLORS, NICE_NAMES
from .data import letterbox, to_tensor
from .models import WasteClassifier, WasteDetector, detections
from .prepare_data import square_crop

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WasteSorter:
    def __init__(self, detector=None, classifier=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        d = torch.load(detector or os.path.join(HERE, "weights", "detector.pt"), map_location="cpu")
        c = torch.load(classifier or os.path.join(HERE, "weights", "classifier.pt"), map_location="cpu")
        self.det = WasteDetector(); self.det.load_state_dict(d["model"]); self.det.to(self.device).eval()
        self.cls = WasteClassifier(); self.cls.load_state_dict(c["model"]); self.cls.to(self.device).eval()
        self.det_size, self.cls_size = d.get("size", 416), c.get("size", 128)
        self.default_conf = float(min(0.6, max(0.25, d.get("conf", 0.3))))

    @torch.no_grad()
    def detect(self, img_bgr, conf=None):
        """Returns boxes [n,5] (x1,y1,x2,y2,score) in pixels of the original photo."""
        h, w = img_bgr.shape[:2]
        boxed, r, ox, oy = letterbox(img_bgr, self.det_size)
        out = self.det(to_tensor(boxed)[None].to(self.device))
        det = detections(out, conf=conf or self.default_conf)[0].numpy()
        det[:, [0, 2]] = ((det[:, [0, 2]] - ox) / r).clip(0, w)
        det[:, [1, 3]] = ((det[:, [1, 3]] - oy) / r).clip(0, h)
        keep = ((det[:, 2] - det[:, 0]) > 4) & ((det[:, 3] - det[:, 1]) > 4)
        return det[keep]

    @torch.no_grad()
    def classify(self, img_bgr, boxes):
        """Returns class probabilities [n,4] for each box."""
        if not len(boxes):
            return np.zeros((0, len(CLASSES)), np.float32)
        crops = [to_tensor(square_crop(img_bgr, b[:4], size=self.cls_size)) for b in boxes]
        x = torch.stack(crops).to(self.device)
        logits = self.cls(x)
        for k in (1, 2, 3):                      # average over 4 rotations: a little more reliable
            logits = logits + self.cls(torch.rot90(x, k, dims=(2, 3)))
        return torch.softmax(logits / 4, dim=1).cpu().numpy()

    def predict(self, img_bgr, conf=None):
        boxes = self.detect(img_bgr, conf)
        probs = self.classify(img_bgr, boxes)
        objects = []
        for b, p in zip(boxes, probs):
            c = int(p.argmax())
            objects.append(dict(box=[int(round(v)) for v in b[:4]], detector_score=float(b[4]),
                                label=CLASSES[c], class_id=c, confidence=float(p[c])))
        objects.sort(key=lambda o: (o["class_id"], o["box"][0]))
        counts = {name: sum(o["class_id"] == i for o in objects) for i, name in enumerate(CLASSES)}
        return dict(objects=objects, counts=counts, image=draw(img_bgr, objects))

    def bins(self, img_bgr, objects, size=160):
        """The "segregated" view: one row of cut-out objects per class."""
        return {name: [square_crop(img_bgr, o["box"], pad=0.06, size=size) for o in objects if o["class_id"] == i]
                for i, name in enumerate(CLASSES)}


def draw(img_bgr, objects):
    """Draw coloured boxes and labels on a copy of the photo."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    t = max(2, round(max(h, w) / 320))
    fs = max(0.45, max(h, w) / 1400)
    for o in objects:
        x1, y1, x2, y2 = o["box"]
        col = COLORS[o["class_id"]][::-1]        # RGB -> BGR
        cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
        text = f"{NICE_NAMES[o['class_id']]} {o['confidence'] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
        ty = y1 - 6 if y1 - th - 8 > 0 else y1 + th + 8
        cv2.rectangle(out, (x1, ty - th - 5), (x1 + tw + 6, ty + 4), col, -1)
        cv2.putText(out, text, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, t // 2), cv2.LINE_AA)
    return out
