"""Shared settings. Change things here, not scattered through the code."""

# The four things we sort. The order matters: it is the class id used everywhere.
CLASSES = ["glass_bottle", "paper_cup", "plastic_bottle", "metal_can"]
NICE_NAMES = ["Glass bottle", "Paper cup", "Plastic bottle", "Metal can"]
# Box colours (R, G, B) for drawing.
COLORS = [(46, 160, 67), (219, 109, 40), (56, 132, 244), (163, 113, 247)]

UNKNOWN = 9          # class id for "an object, but we don't know which of the 4" (detector only)

DET_SIZE = 416       # detector input size (pixels, longest side)
DET_STRIDE = 8       # detector output is 1/8 of the input resolution
CLS_SIZE = 128       # classifier input size
CLS_STORE = 144      # classifier crops are stored a bit larger so we can random-crop

# Kaggle datasets we use (all public, all free). See README for licences.
DATASETS = {
    "drinking": "arkadiyhacks/drinking-waste-classification",
    "vn": "mrgetshjtdone/vn-trash-classification",
    "household": "alistairking/recyclable-and-household-waste-classification",
    "cups": "vencerlanz09/plastic-and-paper-cups-synthetic-image-dataset",
    "taco": "vencerlanz09/taco-dataset-yolo-format",
}
