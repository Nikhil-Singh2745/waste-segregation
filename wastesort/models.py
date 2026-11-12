"""The two CNNs, written from scratch (plain PyTorch layers, random initial weights).

  WasteDetector    finds objects: "there is *something* here, and this is its box"
  WasteClassifier  looks at one cropped object and says which of the 4 classes it is
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from .config import CLASSES, DET_STRIDE


class ConvBN(nn.Sequential):
    """Convolution -> batch-norm -> SiLU activation: the basic building brick."""

    def __init__(self, cin, cout, k=3, stride=1, act=True):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection (as in ResNet)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = ConvBN(cin, cout, 3, stride)
        self.conv2 = ConvBN(cout, cout, 3, act=False)
        self.skip = nn.Identity() if stride == 1 and cin == cout else ConvBN(cin, cout, 1, stride, act=False)

    def forward(self, x):
        return F.silu(self.conv2(self.conv1(x)) + self.skip(x))


def stage(cin, cout, blocks, stride=2):
    return nn.Sequential(ResBlock(cin, cout, stride), *[ResBlock(cout, cout) for _ in range(blocks - 1)])


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)


# ----------------------------------------------------------------------------- classifier
class WasteClassifier(nn.Module):
    """Small ResNet-style classifier. Input 3x128x128 -> 4 class scores. ~2.8M weights."""

    def __init__(self, num_classes=len(CLASSES), width=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            ConvBN(3, w[0], 3, 2),          # 128 -> 64
            ConvBN(w[0], w[0], 3),
            stage(w[0], w[1], 2),           # 64 -> 32
            stage(w[1], w[2], 2),           # 32 -> 16
            stage(w[2], w[3], 2),           # 16 -> 8
            stage(w[3], w[3], 1),           # 8 -> 4
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(w[3], num_classes))
        init_weights(self)

    def forward(self, x):
        return self.head(self.features(x))


# ----------------------------------------------------------------------------- detector
class WasteDetector(nn.Module):
    """Anchor-free single-shot detector in the style of CenterNet. ~3.5M weights.

    A backbone shrinks the image to 1/8, 1/16 and 1/32 size; a small feature pyramid
    merges those back into one 1/8-size map. For every cell of that map the head says:
        channel 0    is an object centred here?   (heat-map, logit)
        channel 1-2  where exactly inside the cell is the centre (0..1)
        channel 3-4  log of box width / height, measured in cells
    It is class-agnostic: it only finds objects, the classifier names them.
    """

    def __init__(self, width=(32, 64, 128, 192, 256), neck=96):
        super().__init__()
        w = width
        self.stem = nn.Sequential(ConvBN(3, w[0], 3, 2), ConvBN(w[0], w[1], 3, 2), ResBlock(w[1], w[1]))  # 1/4
        self.c3 = stage(w[1], w[2], 2)      # 1/8
        self.c4 = stage(w[2], w[3], 2)      # 1/16
        self.c5 = stage(w[3], w[4], 2)      # 1/32
        self.l3, self.l4, self.l5 = ConvBN(w[2], neck, 1), ConvBN(w[3], neck, 1), ConvBN(w[4], neck, 1)
        self.p4, self.p3 = ConvBN(neck, neck, 3), ConvBN(neck, neck, 3)
        self.head = nn.Sequential(ConvBN(neck, neck, 3), ConvBN(neck, neck, 3), nn.Conv2d(neck, 5, 1))
        init_weights(self)
        nn.init.constant_(self.head[-1].bias, 0.0)
        nn.init.constant_(self.head[-1].bias[0:1], -2.19)   # start by predicting "no object" almost everywhere
        nn.init.constant_(self.head[-1].bias[3:5], 1.5)     # start with boxes ~4.5 cells wide

    def forward(self, x):
        c3 = self.c3(self.stem(x))
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        p5 = self.l5(c5)
        p4 = self.p4(self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.p3(self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        return self.head(p3)


def decode_boxes(out, stride=DET_STRIDE):
    """Turn the raw head output [B,5,H,W] into boxes [B,H,W,4] (x1,y1,x2,y2 in input pixels)."""
    b, _, h, w = out.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=out.device), torch.arange(w, device=out.device), indexing="ij")
    cx = (xs + torch.sigmoid(out[:, 1])) * stride
    cy = (ys + torch.sigmoid(out[:, 2])) * stride
    bw = torch.exp(out[:, 3].clamp(-4, 6)) * stride
    bh = torch.exp(out[:, 4].clamp(-4, 6)) * stride
    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)


@torch.no_grad()
def detections(out, conf=0.3, iou=0.5, topk=60, stride=DET_STRIDE):
    """Raw output -> list (one per image) of tensors [n,5] = x1,y1,x2,y2,score."""
    heat = torch.sigmoid(out[:, 0])
    peaks = (F.max_pool2d(heat[:, None], 3, 1, 1)[:, 0] == heat) & (heat >= conf)   # keep local maxima only
    boxes = decode_boxes(out, stride)
    results = []
    for i in range(out.shape[0]):
        s, bx = heat[i][peaks[i]], boxes[i][peaks[i]]
        if len(s) > topk:
            s, idx = s.topk(topk)
            bx = bx[idx]
        keep = nms(bx.float(), s.float(), iou)
        results.append(torch.cat([bx[keep], s[keep, None]], dim=1).float().cpu())
    return results


def count_params(model):
    return sum(p.numel() for p in model.parameters())
