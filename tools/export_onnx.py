"""Export the two trained CNNs to ONNX so the demo can run inside a web browser (see docs/).

    pip install onnx onnxruntime        (only needed for this script)
    python tools/export_onnx.py

Writes docs/models/detector.onnx, docs/models/classifier.onnx and docs/models/meta.json,
then checks that ONNX gives the same numbers as PyTorch. Re-run it after retraining.
Nothing else in the project depends on these files: app.py and predict.py still use weights/*.pt.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from wastesort.config import CLASSES, COLORS, DET_STRIDE, NICE_NAMES  # noqa: E402
from wastesort.data import MEAN, STD  # noqa: E402
from wastesort.infer import WasteSorter  # noqa: E402

OUT = os.path.join(HERE, "docs", "models")


class ClassifierTTA(nn.Module):
    """The classifier exactly as infer.py uses it: average over 4 rotations, then softmax."""

    def __init__(self, cls):
        super().__init__()
        self.cls = cls

    def forward(self, x):
        # torch.rot90 cannot be exported, so the three rotations are written as flips + transposes.
        r1 = x.flip(3).transpose(2, 3)
        r2 = x.flip(2).flip(3)
        r3 = x.transpose(2, 3).flip(3)
        logits = self.cls(x) + self.cls(r1) + self.cls(r2) + self.cls(r3)
        return torch.softmax(logits / 4, dim=1)


def export(model, example, path, dynamic_batch):
    torch.onnx.export(
        model, example, path, input_names=["input"], output_names=["output"], opset_version=17, dynamo=False,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None,
    )
    print(f"wrote {os.path.relpath(path, HERE)}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    import onnxruntime as ort

    os.makedirs(OUT, exist_ok=True)
    s = WasteSorter(device="cpu")
    det, cls = s.det, ClassifierTTA(s.cls).eval()
    det_path, cls_path = os.path.join(OUT, "detector.onnx"), os.path.join(OUT, "classifier.onnx")
    torch.manual_seed(0)
    xd = torch.randn(1, 3, s.det_size, s.det_size)
    xc = torch.randn(5, 3, s.cls_size, s.cls_size)
    export(det, xd, det_path, dynamic_batch=False)
    export(cls, xc[:2], cls_path, dynamic_batch=True)

    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(dict(classes=CLASSES, nice_names=NICE_NAMES, colors=COLORS, mean=list(MEAN), std=list(STD),
                       det_size=s.det_size, det_stride=DET_STRIDE, cls_size=s.cls_size,
                       default_conf=round(s.default_conf, 2)), f, indent=1)

    # Same input through PyTorch and ONNX must give the same output.
    def reference_classifier(x):          # the same lines as WasteSorter.classify
        logits = s.cls(x)
        for k in (1, 2, 3):
            logits = logits + s.cls(torch.rot90(x, k, dims=(2, 3)))
        return torch.softmax(logits / 4, dim=1)

    with torch.no_grad():
        for name, model, x, path in (("detector", det, xd, det_path), ("classifier", reference_classifier, xc, cls_path)):
            want = model(x).numpy()
            got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
            diff = float(np.abs(want - got).max())
            print(f"{name}: largest difference PyTorch vs ONNX = {diff:.2e}")
            assert diff < 1e-3, f"{name} ONNX export does not match PyTorch"
    print("OK")


if __name__ == "__main__":
    main()
