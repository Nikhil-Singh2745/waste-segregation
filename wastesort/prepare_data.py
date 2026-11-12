"""Download the public datasets and turn them into one clean, small dataset.

Output layout (under --out, default ./data):

  det/{train,val,test}/images/*.jpg   photos, longest side <= 512 px
  det/{train,val,test}/labels/*.txt   one line per object: "class cx cy w h" (0..1, YOLO style)
  cls/{train,val,test}/<class>/*.jpg  144x144 crops of single objects
  my_photos handling: anything you put in my_photos/<class>/ is added to cls/train.

Run:  python -m wastesort.prepare_data
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np

from .config import CLASSES, CLS_STORE, DATASETS, UNKNOWN

cv2.setNumThreads(0)
GLASS, CUP, PLASTIC, CAN = 0, 1, 2, 3
IMG_EXT = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------------- helpers
def find_root(name):
    """Return the folder of a Kaggle dataset, downloading it if needed."""
    slug = DATASETS[name]
    for base in ("/kaggle/input", os.environ.get("WASTE_RAW", "")):
        p = os.path.join(base, slug.split("/")[1])
        if base and os.path.isdir(p):
            return p
    import kagglehub
    return kagglehub.dataset_download(slug)


def find_dir(root, name):
    """Find a sub-folder called `name` anywhere under root."""
    for d, subdirs, _ in os.walk(root):
        if name in subdirs:
            return os.path.join(d, name)
    raise FileNotFoundError(f"{name} not found under {root}")


def bucket(key, val=0.1, test=0.1):
    """Stable pseudo-random split based on a hash of `key`."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000 / 1000
    return "test" if h < test else "val" if h < test + val else "train"


def imread(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)   # applies EXIF rotation
    return img


def square_crop(img, box, pad=0.12, size=CLS_STORE):
    """Cut a square around `box` (x1,y1,x2,y2 in pixels) keeping some real context.

    Bottles are long and thin; instead of squashing them we take a square window
    and fill whatever falls outside the photo by repeating the edge pixels.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    side = max(side, 8)
    sx1, sy1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    sx2, sy2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pl, pt = max(0, -sx1), max(0, -sy1)
    pr, pb = max(0, sx2 - w), max(0, sy2 - h)
    crop = img[max(0, sy1):min(h, sy2), max(0, sx1):min(w, sx2)]
    if pl or pt or pr or pb:
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def read_yolo(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            t = line.split()
            if len(t) >= 5:
                rows.append((int(float(t[0])), *[float(v) for v in t[1:5]]))
    return rows


# ----------------------------------------------------------------------------- workers
def work(job):
    """One job = one source image. Runs in a worker process."""
    kind, src, split, name, boxes, out = job
    try:
        img = imread(src)
        if img is None:
            return Counter(bad=1)
        h, w = img.shape[:2]
        stats = Counter()
        if kind in ("det", "det+cls"):
            scale = min(1.0, 512 / max(h, w))
            small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
            cv2.imwrite(f"{out}/det/{split}/images/{name}.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(f"{out}/det/{split}/labels/{name}.txt", "w") as f:
                for c, cx, cy, bw, bh in boxes:
                    f.write(f"{c} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n")
            stats[f"det/{split}"] += 1
        if kind == "det+cls":
            for i, (c, cx, cy, bw, bh) in enumerate(boxes):
                if c == UNKNOWN or bw * w < 12 or bh * h < 12:
                    continue
                box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
                cv2.imwrite(f"{out}/cls/{split}/{CLASSES[c]}/{name}_{i}.jpg", square_crop(img, box),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                stats[f"cls/{split}/{CLASSES[c]}"] += 1
        if kind == "cls":
            c = boxes
            crop = square_crop(img, (0, 0, w, h), pad=0.0)
            cv2.imwrite(f"{out}/cls/{split}/{CLASSES[c]}/{name}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            stats[f"cls/{split}/{CLASSES[c]}"] += 1
        return stats
    except Exception as e:  # a broken file should not kill the whole run
        return Counter({f"error:{type(e).__name__}": 1})


# ----------------------------------------------------------------------------- sources
def jobs_drinking(out):
    """Real phone photos of cans / glass bottles / plastic bottles with boxes."""
    d = find_dir(find_root("drinking"), "YOLO_imgs")
    remap = {0: CAN, 1: GLASS, 2: PLASTIC, 3: PLASTIC}   # AluCan, Glass, HDPE milk bottle, PET
    jobs = []
    for p in sorted(glob.glob(d + "/*")):
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() not in IMG_EXT:
            continue
        boxes = [(remap[c], *r) for c, *r in read_yolo(os.path.join(d, stem + ".txt"))]
        if not boxes:
            continue
        # Photos were taken in sequences (same object, slightly different angle), so a
        # random split would put near-identical photos in train and test. Split in
        # blocks of 20 consecutive photos instead.
        m = re.match(r"([A-Za-z]+)([\d,]+)", stem)
        prefix, num = m.group(1), int(m.group(2).replace(",", ""))
        split = bucket(f"{prefix}{num // 20}")
        if prefix == "mix":      # the few real photos with several different items: keep ALL for testing
            split = "test"
        jobs.append(("det+cls", p, split, "dw_" + stem.replace(",", ""), boxes, out))
    return jobs


def jobs_vn(out):
    """Web photos: paper cups, cans, PET bottles (classification only)."""
    root = find_dir(find_root("vn"), "VN_trash_classification")
    jobs = []
    for folder, cls in (("Paper_cup", CUP), ("Alu", CAN), ("PET", PLASTIC)):
        for sp in ("Train", "Test"):
            for p in sorted(glob.glob(f"{root}/{sp}/{folder}/*")):
                stem, ext = os.path.splitext(os.path.basename(p))
                if ext.lower() not in IMG_EXT:
                    continue
                # this dataset re-uses photos from the drinking-waste set: skip those copies
                if re.match(r"(AluCan|PET|Glass|HDPE)", stem):
                    continue
                jobs.append(("cls", p, bucket("vn" + folder + stem), f"vn_{folder}_{sp}_{stem}"[:90].replace(" ", "_"), cls, out))
    return jobs


def jobs_household(out):
    """Web photos of glass bottles (the VN set has none)."""
    root = find_dir(find_root("household"), "glass_beverage_bottles")
    jobs = []
    for p in sorted(glob.glob(root + "/*/*")):
        if not p.lower().endswith(IMG_EXT):
            continue
        sub = os.path.basename(os.path.dirname(p))
        stem = os.path.splitext(os.path.basename(p))[0]
        jobs.append(("cls", p, bucket("hh" + sub + stem), f"hh_glass_{sub}_{stem}".replace(" ", "_"), GLASS, out))
    return jobs


def jobs_cups(out, limit=3000):
    """Computer-generated cup scenes (cups pasted on random backgrounds) with boxes.

    They teach the detector what cups look like, and every second image also gives the
    classifier one cup crop on a busy background, so it cannot cheat by learning
    "white studio background = paper cup".
    """
    root = find_root("cups")
    ann_file = glob.glob(root + "/**/coco_instances.json", recursive=True)[0]
    coco = json.load(open(ann_file))
    img_dir = os.path.join(os.path.dirname(ann_file), "images_raw")
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a["bbox"])
    jobs = []
    for n, im in enumerate(coco["images"][:limit]):
        w, h = im["width"], im["height"]
        raw = sorted((b for b in by_img.get(im["id"], []) if b[2] > 8 and b[3] > 8), key=lambda b: -b[2] * b[3])
        boxes = [(UNKNOWN, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h) for x, y, bw, bh in raw]
        if not boxes:
            continue
        kind = "det"
        if n % 2 == 0 and min(raw[0][2], raw[0][3]) > 60:   # biggest cup -> classifier crop
            boxes[0] = (CUP, *boxes[0][1:])
            kind = "det+cls"
        stem = os.path.splitext(im["file_name"])[0]
        jobs.append((kind, os.path.join(img_dir, im["file_name"]), bucket("cups" + stem, 0.05, 0.05), "sc_" + stem, boxes, out))
    return jobs


def jobs_taco(out):
    """Litter photographed outdoors. We keep only bottles / cans / cups that are big enough."""
    root = find_root("taco")
    keep = {2, 4, 7}   # Bottle, Can, Cup
    jobs = []
    for sp, ours in (("train", "train"), ("valid", "val"), ("test", "test")):
        d = find_dir(root, sp) if not os.path.isdir(os.path.join(root, sp)) else os.path.join(root, sp)
        for p in sorted(glob.glob(d + "/images/*.jpg")):
            stem = os.path.splitext(os.path.basename(p))[0]
            rows = read_yolo(f"{d}/labels/{stem}.txt")
            # skip photos that also contain other big litter we would have to call "background"
            if any(c not in keep and min(bw, bh) > 0.08 for c, _, _, bw, bh in rows):
                continue
            boxes = [(UNKNOWN, *r) for c, *r in rows if c in keep and min(r[2], r[3]) > 0.08]
            if boxes:
                jobs.append(("det", p, ours, "tc_" + hashlib.md5(stem.encode()).hexdigest()[:12], boxes, out))
    return jobs


def jobs_my_photos(out, folder):
    """Your own photos: my_photos/<class>/*.jpg -> added to classifier training."""
    jobs = []
    for c, name in enumerate(CLASSES):
        for p in sorted(glob.glob(f"{folder}/{name}/*")):
            if p.lower().endswith(IMG_EXT):
                stem = os.path.splitext(os.path.basename(p))[0]
                jobs.append(("cls", p, "train", f"my_{stem}".replace(" ", "_"), c, out))
    return jobs


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--my-photos", default="my_photos")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    out = args.out
    if os.path.isdir(out):
        shutil.rmtree(out)
    for sp in ("train", "val", "test"):
        os.makedirs(f"{out}/det/{sp}/images")
        os.makedirs(f"{out}/det/{sp}/labels")
        for c in CLASSES:
            os.makedirs(f"{out}/cls/{sp}/{c}")

    jobs = []
    for fn in (jobs_drinking, jobs_vn, jobs_household, jobs_cups, jobs_taco):
        j = fn(out)
        print(f"{fn.__name__[5:]:10s} {len(j):6d} source images")
        jobs += j
    mine = jobs_my_photos(out, args.my_photos)
    if mine:
        print(f"{'my_photos':10s} {len(mine):6d} source images")
        jobs += mine

    total = Counter()
    with Pool(args.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(work, jobs, chunksize=32)):
            total.update(st)
            if i % 2000 == 0:
                print(f"  processed {i}/{len(jobs)}", flush=True)
    print("\nFinished. Counts:")
    for k in sorted(total):
        print(f"  {k:32s} {total[k]}")
    json.dump(dict(total), open(f"{out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
