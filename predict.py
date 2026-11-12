"""Sort one photo from the command line.

    python predict.py my_pile.jpg            -> saves my_pile_sorted.jpg and prints the counts
    python predict.py my_pile.jpg --conf 0.4 -> stricter (fewer, surer boxes)
"""
import argparse
import os

import cv2

from wastesort.config import CLASSES, NICE_NAMES
from wastesort.infer import WasteSorter

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--conf", type=float, default=None, help="detector confidence threshold (0-1)")
args = ap.parse_args()

img = cv2.imread(args.image)
if img is None:
    raise SystemExit(f"Could not read {args.image}")
result = WasteSorter().predict(img, args.conf)
out = os.path.splitext(args.image)[0] + "_sorted.jpg"
cv2.imwrite(out, result["image"])
print(f"Found {len(result['objects'])} objects:")
for name, nice in zip(CLASSES, NICE_NAMES):
    print(f"  {nice:15s} {result['counts'][name]}")
for o in result["objects"]:
    print(f"  - {o['label']:15s} {o['confidence'] * 100:5.1f}%  box={o['box']}")
print("Saved", out)
