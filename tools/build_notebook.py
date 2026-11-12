"""Builds notebooks/train_on_kaggle.ipynb from the source files, so the notebook is fully
self-contained (upload ONE file to Kaggle/Colab, press Run All).

Run after changing any code:  python tools/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["wastesort/__init__.py", "wastesort/config.py", "wastesort/models.py", "wastesort/data.py",
         "wastesort/prepare_data.py", "wastesort/train_classifier.py", "wastesort/train_detector.py",
         "wastesort/infer.py", "wastesort/evaluate.py"]

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                               "source": s.strip("\n").splitlines(True)})

md("""
# Waste Sorter - training notebook

Trains two CNNs **from scratch** (no pretrained weights):

1. a **detector** that draws a box around every item in a photo
2. a **classifier** that names each boxed item: glass bottle / paper cup / plastic bottle / metal can

**Before you press Run All** (Kaggle, right-hand panel -> *Session options*):
- **Accelerator: GPU T4 x2** (or P100)
- **Internet: On**

Total time is roughly 2 hours on a free GPU. At the end, download **`waste_sorter_weights.zip`**
from the *Output* panel and unzip it into the project folder on your laptop.
""")
code("""
import os, torch
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/kaggle/working/waste-sorter" if ON_KAGGLE else "/content/waste-sorter"
os.makedirs(WORK + "/wastesort", exist_ok=True)
os.chdir(WORK)
print("Working in", WORK)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - turn the GPU on, or this will take days!")
""")
md("## 1. The code\nThe next cells just write the project's Python files to disk.")
for f in FILES:
    code(f"%%writefile {f}\n" + open(os.path.join(ROOT, f)).read())
md("""
## 2. Download and prepare the data
Five public Kaggle datasets are downloaded and converted into one small, clean dataset (about 3 minutes).

*Optional:* to add your own photos, upload them as a Kaggle dataset / to Colab and copy them into
`my_photos/<class>/` (`glass_bottle`, `paper_cup`, `plastic_bottle`, `metal_can`) before running this cell.
""")
code("!pip -q install kagglehub\n!python -m wastesort.prepare_data --out data")
code("""
import glob, random, cv2, matplotlib.pyplot as plt
from wastesort.config import CLASSES
fig, ax = plt.subplots(4, 10, figsize=(16, 7))
for r, c in enumerate(CLASSES):
    for k, f in enumerate(random.sample(glob.glob(f"data/cls/train/{c}/*.jpg"), 10)):
        ax[r, k].imshow(cv2.imread(f)[:, :, ::-1]); ax[r, k].axis("off")
    ax[r, 0].set_title(c, loc="left")
plt.tight_layout(); plt.show()
""")
md("## 3. Train the classifier (about 15 minutes)")
code("!python -m wastesort.train_classifier --data data --epochs 60")
code("""
from IPython.display import Image, display
display(Image("runs/classifier/curves.png")); display(Image("runs/classifier/confusion_matrix.png"))
""")
md("## 4. Train the detector (about 1.5 hours)")
code("!python -m wastesort.train_detector --data data --epochs 80")
code('display(Image("runs/detector/curves.png"))')
md("## 5. Test the whole pipeline on photos it has never seen")
code("!python -m wastesort.evaluate --data data")
code("""
import glob
fig, ax = plt.subplots(3, 4, figsize=(16, 12))
for a, f in zip(ax.ravel(), sorted(glob.glob("runs/pipeline/example_*.jpg"))[-12:]):
    a.imshow(cv2.imread(f)[:, :, ::-1]); a.axis("off")
plt.tight_layout(); plt.show()
""")
md("## 6. Pack the results\nDownload `waste_sorter_weights.zip` (Kaggle: *Output* panel on the right, after the run finishes).")
code("""
import shutil
os.makedirs("export", exist_ok=True)
shutil.copytree("weights", "export/weights", dirs_exist_ok=True)
shutil.copytree("runs", "export/runs", dirs_exist_ok=True)
zip_path = shutil.make_archive(("/kaggle/working" if ON_KAGGLE else "/content") + "/waste_sorter_weights", "zip", "export")
print("Saved", zip_path)
if not ON_KAGGLE:
    from google.colab import files; files.download(zip_path)
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}}
out = os.path.join(ROOT, "notebooks", "train_on_kaggle.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, f"({len(cells)} cells)")
