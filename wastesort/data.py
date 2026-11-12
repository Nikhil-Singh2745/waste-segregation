"""Datasets and augmentation (the random changes applied to training images)."""
import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from .config import CLASSES, CLS_SIZE, DET_SIZE, DET_STRIDE

cv2.setNumThreads(0)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
GRAY = 114


def to_tensor(img_bgr):
    """uint8 BGR image (OpenCV) -> normalised float tensor [3,H,W]."""
    x = torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1])).permute(2, 0, 1).float() / 255
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


# ----------------------------------------------------------------------------- classifier
class Rot90:
    """Rotate by 0/90/180/270 degrees: waste lies in any direction on the floor."""

    def __call__(self, x):
        return torch.rot90(x, random.randint(0, 3), dims=(-2, -1))


def cls_transform(train, size=CLS_SIZE):
    norm = [T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)]
    if not train:
        return T.Compose([T.Resize((size, size), antialias=True), *norm])
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.45, 1.0), ratio=(0.75, 1.33), antialias=True),
        T.RandomHorizontalFlip(),
        Rot90(),
        T.RandomApply([T.RandomRotation(20)], p=0.3),
        T.ColorJitter(0.35, 0.35, 0.35, 0.04),
        T.RandomGrayscale(0.05),
        *norm,
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


class ClsDataset(Dataset):
    """Reads data/cls/<split>/<class>/*.jpg"""

    def __init__(self, root, split, train=False, size=CLS_SIZE, limit=None):
        self.items = []
        for c, name in enumerate(CLASSES):
            files = sorted(glob.glob(f"{root}/cls/{split}/{name}/*.jpg"))
            self.items += [(f, c) for f in files]
        if limit:
            random.Random(0).shuffle(self.items)
            self.items = self.items[:limit]
        self.tf = cls_transform(train, size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, c = self.items[i]
        img = cv2.imread(path)[:, :, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return self.tf(x), c

    def class_counts(self):
        return np.bincount([c for _, c in self.items], minlength=len(CLASSES))


# ----------------------------------------------------------------------------- detector
def letterbox(img, size=DET_SIZE, scale=1.0, center=True, rng=None):
    """Resize keeping the aspect ratio and pad to size x size with gray.

    Returns (new image, zoom factor, x offset, y offset)."""
    h, w = img.shape[:2]
    r = size / max(h, w) * scale
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), GRAY, np.uint8)
    if center:
        ox, oy = (size - nw) // 2, (size - nh) // 2
    else:
        ox, oy = rng.randint(0, size - nw), rng.randint(0, size - nh)
    out[oy:oy + nh, ox:ox + nw] = small
    return out, r, ox, oy


def cover(img, boxes, ow, oh, rng):
    """Zoom the image so it fills an ow x oh window, then cut a random window out of it."""
    h, w = img.shape[:2]
    r = max(ow / w, oh / h) * rng.uniform(1.0, 1.25)
    nw, nh = max(ow, round(w * r)), max(oh, round(h * r))
    big = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    x0, y0 = rng.randint(0, nw - ow), rng.randint(0, nh - oh)
    b = boxes * [nw / w, nh / h, nw / w, nh / h] - [x0, y0, x0, y0] if len(boxes) else boxes
    return big[y0:y0 + oh, x0:x0 + ow], b


def clip_boxes(boxes, w, h, min_visible=0.45, min_side=6):
    """Clip boxes to the image; drop the ones that are mostly cut off or tiny."""
    if not len(boxes):
        return boxes.reshape(0, 4), np.zeros(0, bool)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    c = boxes.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, w)
    c[:, [1, 3]] = c[:, [1, 3]].clip(0, h)
    cw, ch = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    keep = (cw * ch > min_visible * area) & (cw >= min_side) & (ch >= min_side)
    return c[keep], keep


def color_jitter(img, rng):
    """Random brightness / contrast / saturation / tint."""
    x = img.astype(np.float32)
    x = x * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.6, 1.4)
    x = x * (1 + np.array([rng.uniform(-0.06, 0.06) for _ in range(3)], np.float32))
    return x.clip(0, 255).astype(np.uint8)


class DetDataset(Dataset):
    """Reads data/det/<split>/images + labels. Returns image tensor and pixel boxes.

    Training augmentation builds "pile" pictures out of single-object photos:
      * mosaic  : four photos tiled into one
      * paste   : extra objects cut from other photos and blended in next to existing ones
      * plus flips, 90-degree turns, zoom and colour changes
    """

    def __init__(self, root, split, train=False, size=DET_SIZE, limit=None, mosaic=0.5, paste=0.35):
        self.files = sorted(glob.glob(f"{root}/det/{split}/images/*.jpg"))
        if limit:
            random.Random(0).shuffle(self.files)
            self.files = self.files[:limit]
        self.pasteable = [i for i, f in enumerate(self.files) if os.path.basename(f).startswith("dw_")]
        self.train, self.size, self.mosaic, self.paste = train, size, mosaic, paste

    def __len__(self):
        return len(self.files)

    def load(self, i):
        path = self.files[i]
        img = cv2.imread(path)
        h, w = img.shape[:2]
        rows = []
        lab = path.replace("/images/", "/labels/")[:-4] + ".txt"
        for line in open(lab):
            t = line.split()
            rows.append([float(v) for v in t[:5]])
        rows = np.array(rows, np.float32).reshape(-1, 5)
        boxes = np.stack([(rows[:, 1] - rows[:, 3] / 2) * w, (rows[:, 2] - rows[:, 4] / 2) * h,
                          (rows[:, 1] + rows[:, 3] / 2) * w, (rows[:, 2] + rows[:, 4] / 2) * h], axis=1)
        return img, boxes, rows[:, 0].astype(np.int64)

    def paste_objects(self, canvas, boxes, rng):
        """Blend 1-3 objects from other photos into the picture, soft-edged."""
        S = self.size
        for _ in range(rng.randint(1, 3)):
            img, bxs, _ = self.load(rng.choice(self.pasteable))
            if not len(bxs):
                continue
            x1, y1, x2, y2 = bxs[rng.randrange(len(bxs))]
            bw, bh = x2 - x1, y2 - y1
            m = 0.10 * max(bw, bh)                                    # margin of real floor around it
            px1, py1 = int(max(0, x1 - m)), int(max(0, y1 - m))
            px2, py2 = int(min(img.shape[1], x2 + m)), int(min(img.shape[0], y2 + m))
            patch = img[py1:py2, px1:px2]
            if min(patch.shape[:2]) < 16:
                continue
            k = rng.randint(0, 3)
            patch = np.ascontiguousarray(np.rot90(patch, k))
            ob = np.array([x1 - px1, y1 - py1, x2 - px1, y2 - py1], np.float32)   # object box inside the patch
            ph0, pw0 = py2 - py1, px2 - px1
            for _ in range(k):                                         # rotate the box with the patch
                ob = np.array([ob[1], pw0 - ob[2], ob[3], pw0 - ob[0]], np.float32)
                pw0, ph0 = ph0, pw0
            r = rng.uniform(0.18, 0.45) * S / max(patch.shape[:2])
            pw, ph = max(8, round(patch.shape[1] * r)), max(8, round(patch.shape[0] * r))
            patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
            ob = ob * r
            for _try in range(8):
                ox, oy = rng.randint(0, S - pw), rng.randint(0, S - ph)
                nb = ob + [ox, oy, ox, oy]
                if len(boxes):                                         # do not bury existing objects
                    ix = (np.minimum(boxes[:, 2], nb[2]) - np.maximum(boxes[:, 0], nb[0])).clip(0)
                    iy = (np.minimum(boxes[:, 3], nb[3]) - np.maximum(boxes[:, 1], nb[1])).clip(0)
                    frac = ix * iy / ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + 1e-6)
                    if frac.max() > 0.25:
                        continue
                yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
                d = np.sqrt(((xx - pw / 2) / (pw / 2)) ** 2 + ((yy - ph / 2) / (ph / 2)) ** 2)
                alpha = np.clip((1.18 - d) / 0.22, 0, 1)[:, :, None]   # solid in the middle, fades at the rim
                region = canvas[oy:oy + ph, ox:ox + pw]
                canvas[oy:oy + ph, ox:ox + pw] = (alpha * patch + (1 - alpha) * region).astype(np.uint8)
                boxes = np.concatenate([boxes, nb[None]]).astype(np.float32)
                break
        return canvas, boxes

    def __getitem__(self, i):
        S = self.size
        if not self.train:
            img, boxes, _ = self.load(i)
            img, r, ox, oy = letterbox(img, S)
            boxes = boxes * r + [ox, oy, ox, oy]
            return to_tensor(img), torch.from_numpy(boxes.astype(np.float32))

        rng = random.Random(torch.randint(0, 2**31, (1,)).item())
        if rng.random() < self.mosaic:
            canvas = np.full((S, S, 3), GRAY, np.uint8)
            cx, cy = int(S * rng.uniform(0.35, 0.65)), int(S * rng.uniform(0.35, 0.65))
            parts = []
            for q, j in enumerate([i] + [rng.randrange(len(self)) for _ in range(3)]):
                x0, y0 = (0, cx)[q % 2], (0, cy)[q // 2]
                ow, oh = (cx, S - cx)[q % 2], (cy, S - cy)[q // 2]
                img, b, _ = self.load(j)
                tile, b = cover(img, b, ow, oh, rng)
                b, _ = clip_boxes(b, ow, oh)
                canvas[y0:y0 + oh, x0:x0 + ow] = tile
                parts.append(b + [x0, y0, x0, y0])
            img, boxes = canvas, np.concatenate(parts).astype(np.float32)
        else:
            img, boxes, _ = self.load(i)
            img, r, ox, oy = letterbox(img, S, scale=rng.uniform(0.55, 1.0), center=False, rng=rng)
            boxes = (boxes * r + [ox, oy, ox, oy]).astype(np.float32)
        if self.pasteable and rng.random() < self.paste:
            img, boxes = self.paste_objects(img, boxes, rng)
        if rng.random() < 0.5:                                         # left-right flip
            img = img[:, ::-1]
            boxes = np.stack([S - boxes[:, 2], boxes[:, 1], S - boxes[:, 0], boxes[:, 3]], 1) if len(boxes) else boxes
        if rng.random() < 0.5:                                         # up-down flip
            img = img[::-1]
            boxes = np.stack([boxes[:, 0], S - boxes[:, 3], boxes[:, 2], S - boxes[:, 1]], 1) if len(boxes) else boxes
        if rng.random() < 0.5:                                         # quarter turn (swap x and y)
            img = img.transpose(1, 0, 2)
            boxes = boxes[:, [1, 0, 3, 2]] if len(boxes) else boxes
        img = color_jitter(np.ascontiguousarray(img), rng)
        return to_tensor(img), torch.from_numpy(np.asarray(boxes, np.float32).reshape(-1, 4))


def det_collate(batch):
    """Stack images and build the training targets for the detector.

    For each object we mark the grid cell that contains its centre as "positive" and paint
    a soft blob (Gaussian) around it on the heat-map, so near-misses are punished less.
    """
    imgs = torch.stack([b[0] for b in batch])
    B, _, H, W = imgs.shape
    h, w = H // DET_STRIDE, W // DET_STRIDE
    heat = torch.zeros(B, h, w)
    pos = []            # rows: batch index, cell y, cell x, x1, y1, x2, y2
    ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    for bi, (_, boxes) in enumerate(batch):
        for x1, y1, x2, y2 in boxes.tolist():
            cx, cy = (x1 + x2) / 2 / DET_STRIDE, (y1 + y2) / 2 / DET_STRIDE
            ix, iy = min(w - 1, max(0, int(cx))), min(h - 1, max(0, int(cy)))
            sx = max(0.8, (x2 - x1) / DET_STRIDE / 6)
            sy = max(0.8, (y2 - y1) / DET_STRIDE / 6)
            g = torch.exp(-((xs - ix) ** 2 / (2 * sx * sx) + (ys - iy) ** 2 / (2 * sy * sy)))
            heat[bi] = torch.maximum(heat[bi], g)
            pos.append([bi, iy, ix, x1, y1, x2, y2])
    pos = torch.tensor(pos, dtype=torch.float32).reshape(-1, 7)
    return imgs, heat, pos, [b[1] for b in batch]
