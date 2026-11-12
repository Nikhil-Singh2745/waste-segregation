# Waste Sorter

Take a photo of mixed waste. The program draws a box around every item and sorts each one
into a bin: **glass bottle**, **paper cup**, **plastic bottle** or **metal can**.

Everything is a CNN trained **from scratch**. No pretrained weights, no YOLO, no downloaded
models: the networks are defined layer by layer in `wastesort/models.py` and start from random numbers.

## How it works

```
photo ──> [1] DETECTOR CNN ──> boxes ──> crop each box ──> [2] CLASSIFIER CNN ──> label per box
          "where are the items?"                            "what is this item?"
```

|        | Detector                                                                                                                                                     | Classifier                                                                            |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| Job    | find items, draw boxes (does not care what they are)                                                                                                         | name one cropped item                                                                 |
| Design | CenterNet-style, anchor-free: ResNet-like backbone + small feature pyramid; for every cell of a 52x52 grid it predicts "is an item centred here?" + box size | small ResNet: 5 stages of residual blocks, global average pooling, dropout, 4 outputs |
| Input  | 416x416 photo                                                                                                                                                | 128x128 crop                                                                          |
| Size   | 4.5 million weights                                                                                                                                          | 4.0 million weights                                                                   |
| Loss   | focal loss (heat-map) + GIoU (box) + L1 (centre offset)                                                                                                      | cross-entropy with label smoothing and class weights                                  |

Why two networks instead of one? The classifier can learn from the many photos that only
have a label (no box), which is where almost all paper-cup photos come from. It also gives
you two clean numbers for a report: "how many items did we find" and "how many did we name correctly".

## Run locally after training

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Open http://127.0.0.1:7860 in a browser, upload or snap a photo, press **Sort the waste**.

Without the web page:

```bash
python predict.py my_photo.jpg
```

## Data

`wastesort/prepare_data.py` downloads these public Kaggle datasets and merges them:

| Dataset                                                                                                                         | What we take from it                                                      | Licence      |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------ |
| [Drinking Waste Classification](https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification)                     | 4,800 phone photos of cans, glass bottles, plastic bottles **with boxes** | CC0          |
| [VN Trash Classification](https://www.kaggle.com/datasets/mrgetshjtdone/vn-trash-classification)                                | paper cups, cans, PET bottles (label only)                                | MIT          |
| [Recyclable and Household Waste](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)    | glass beverage bottles (label only)                                       | MIT          |
| [Plastic and Paper Cups Synthetic](https://www.kaggle.com/datasets/vencerlanz09/plastic-and-paper-cups-synthetic-image-dataset) | 3,000 computer-generated cup scenes **with boxes**                        | CC BY-SA 4.0 |
| [TACO (YOLO format)](https://www.kaggle.com/datasets/vencerlanz09/taco-dataset-yolo-format)                                     | ~600 outdoor litter photos with bottles/cans/cups **with boxes**          | CC BY 4.0    |

Because nearly all source photos show a single item, training builds artificial "piles":
four photos are tiled into one (mosaic) and extra items are cut from other photos and blended in.

Photos are split into train / validation / test (about 80 / 10 / 10). The drinking-waste photos were
shot in bursts, so they are split in blocks of consecutive photos to keep near-identical shots out of the test set.
The 24 real photos that contain several different items are all kept for testing.

## Results

These numbers come from the full training run. The trained weights ship in `weights/`
so the demo works straight away. All numbers are measured on test photos never used for training.

| Full run                                            | Result                                        |
| --------------------------------------------------- | --------------------------------------------- |
| Classifier accuracy (test crops)                    | **89.2%** (89.6% with test-time augmentation) |
| Detector AP@0.5 / precision / recall                | 0.97 / 0.97 / 0.94                            |
| Classifier accuracy on the items the detector found | 92.6%                                         |
| **End-to-end: item found AND named correctly**      | **90.7%**                                     |

| Class          | Items | End-to-end success |
| -------------- | ----- | ------------------ |
| Plastic bottle | 250   | 98.0%              |
| Glass bottle   | 224   | 95.5%              |
| Paper cup      | 81    | 92.6%              |
| Metal can      | 188   | 74.5%              |

![example](docs/example.jpg)

What the results show honestly:

- Boxes on real multi-item photos are very good: the detector finds 94% of items.
- **Metal can is the weak class.** It is mostly confused with plastic bottles (blue/white cans look like bottles).
- Paper cup looks strong, but its test items are mostly computer-generated scenes, and no free dataset has real
  photos of used paper cups on a floor. Expect lower accuracy on your own cup photos; adding them (see above) helps.

The full numbers are in `runs/classifier/results.json`, `runs/detector/results.json`
and `runs/pipeline/report.txt`, with plots next to them.

## Files

```
app.py                      web demo
predict.py                  command-line demo
notebooks/original.ipynb   the original notebook before training
wastesort/config.py         class names, image sizes, dataset list
wastesort/models.py         the two CNNs
wastesort/data.py           loading + augmentation (mosaic, paste, flips, colour)
wastesort/prepare_data.py   downloads and merges the datasets
wastesort/train_classifier.py / train_detector.py   training loops
wastesort/evaluate.py       tests detector + classifier together
wastesort/infer.py          the pipeline used by the demos
tools/build_notebook.py     rebuilds the notebook after you change code
weights/                    trained models go here
```

To retrain on your own machine instead:

```bash
python -m wastesort.prepare_data
```

```bash
python -m wastesort.train_classifier --epochs 60
```

```bash
python -m wastesort.train_detector --epochs 80
```

```bash
python -m wastesort.evaluate
```
