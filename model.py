"""
model.py — Arsitektur CenterNet (ResNet-50) + decode + inferensi.
HARUS identik dengan arsitektur saat training agar best.pth bisa dimuat.
Memakai PIL (bukan OpenCV) agar ringan & mudah di-deploy.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------
# Konstanta
# ----------------------------------------------------------------------------
CLASSES = [
    "Cymodocea rotundata",
    "Enhalus acoroides",
    "Halodule uninervis",
    "Halophila ovalis",
    "Syringodium isoetifolium",
    "Thalassia hemprichii",
]
NUM_CLASSES = len(CLASSES)
INPUT_SIZE = 512
DOWN_RATIO = 4
TOPK = 300

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# warna per kelas (RGB) untuk bbox
CLASS_COLORS = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 200, 200),
]

# ambang confidence per-kelas (titik F1-optimal dari data valid)
DEFAULT_THRESH = {
    "Cymodocea rotundata": 0.215,
    "Enhalus acoroides": 0.203,
    "Halodule uninervis": 0.176,
    "Halophila ovalis": 0.305,
    "Syringodium isoetifolium": 0.235,
    "Thalassia hemprichii": 0.216,
}


# ----------------------------------------------------------------------------
# Arsitektur
# ----------------------------------------------------------------------------
class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet50(weights=None)  # bobot datang dari best.pth
        self.conv1 = net.conv1; self.bn1 = net.bn1
        self.relu = net.relu; self.maxpool = net.maxpool
        self.layer1 = net.layer1; self.layer2 = net.layer2
        self.layer3 = net.layer3; self.layer4 = net.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return x


class CenterNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.deconv = self._make_deconv([256, 128, 64], [2048, 256, 128])
        hi = 64
        self.hm_head = nn.Sequential(nn.Conv2d(hi, 64, 3, padding=1), nn.ReLU(True),
                                     nn.Conv2d(64, num_classes, 1))
        self.off_head = nn.Sequential(nn.Conv2d(hi, 64, 3, padding=1), nn.ReLU(True),
                                      nn.Conv2d(64, 2, 1))
        self.wh_head = nn.Sequential(nn.Conv2d(hi, 64, 3, padding=1), nn.ReLU(True),
                                     nn.Conv2d(64, 2, 1))

    def _make_deconv(self, out_chs, in_chs):
        layers = []
        for ic, oc in zip(in_chs, out_chs):
            layers += [nn.ConvTranspose2d(ic, oc, 4, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(oc), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.deconv(feat)
        return {"hm": self.hm_head(feat), "reg": self.off_head(feat),
                "wh": self.wh_head(feat)}


# ----------------------------------------------------------------------------
# Decode
# ----------------------------------------------------------------------------
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    return heat * (hmax == heat).float()


def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return _gather_feat(feat, ind)


def _topk(scores, K):
    B, C, H, W = scores.size()
    ts, ti = torch.topk(scores.view(B, C, -1), K)
    ti = ti % (H * W)
    ys = (ti // W).float(); xs = (ti % W).float()
    ts2, ti2 = torch.topk(ts.view(B, -1), K)
    clses = (ti2 // K).int()
    ti = _gather_feat(ti.view(B, -1, 1), ti2).view(B, K)
    ys = _gather_feat(ys.view(B, -1, 1), ti2).view(B, K)
    xs = _gather_feat(xs.view(B, -1, 1), ti2).view(B, K)
    return ts2, ti, clses, ys, xs


@torch.no_grad()
def decode(out, K=TOPK):
    hm = torch.sigmoid(out["hm"]); hm = _nms(hm)
    B = hm.size(0)
    scores, inds, clses, ys, xs = _topk(hm, K)
    reg = _transpose_and_gather_feat(out["reg"], inds)
    xs = xs.view(B, K, 1) + reg[:, :, 0:1]
    ys = ys.view(B, K, 1) + reg[:, :, 1:2]
    wh = _transpose_and_gather_feat(out["wh"], inds)
    x1 = xs - wh[:, :, 0:1] / 2; y1 = ys - wh[:, :, 1:2] / 2
    x2 = xs + wh[:, :, 0:1] / 2; y2 = ys + wh[:, :, 1:2] / 2
    boxes = torch.cat([x1, y1, x2, y2], dim=2) * DOWN_RATIO
    return (boxes[0].cpu().numpy(), scores[0].view(-1).cpu().numpy(),
            clses[0].view(-1).cpu().numpy())


# ----------------------------------------------------------------------------
# Loader & inferensi
# ----------------------------------------------------------------------------
def load_model(ckpt_path, device="cpu"):
    torch.set_num_threads(1)  # hemat memori di server kecil
    model = CenterNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def load_thresholds(json_path=None):
    if json_path:
        try:
            with open(json_path) as f:
                saved = json.load(f)
            return saved.get("per_class", DEFAULT_THRESH)
        except Exception:
            pass
    return DEFAULT_THRESH


@torch.no_grad()
def predict(model, pil_img, per_class_thresh, device="cpu"):
    """pil_img: PIL.Image RGB.
       Return (dets, hm_combined) — dets list deteksi, hm_combined array 0..1 (HxW)."""
    img = pil_img.convert("RGB")
    W0, H0 = img.size
    resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    out = model(tensor)
    boxes, scores, clses = decode(out)
    hm = torch.sigmoid(out["hm"])[0].cpu().numpy()      # (C, 128, 128)
    hm_combined = hm.max(axis=0)                         # (128, 128) 0..1

    thr_vec = np.array([per_class_thresh[CLASSES[c]] for c in range(NUM_CLASSES)],
                       dtype=np.float32)
    keep = scores >= thr_vec[clses.astype(int)]
    boxes, scores, clses = boxes[keep], scores[keep], clses[keep]

    boxes[:, [0, 2]] *= W0 / INPUT_SIZE
    boxes[:, [1, 3]] *= H0 / INPUT_SIZE
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W0)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H0)

    dets = []
    order = np.argsort(-scores)
    for i in order:
        c = int(clses[i])
        dets.append({"cls": c, "name": CLASSES[c], "score": float(scores[i]),
                     "box": [float(v) for v in boxes[i]],
                     "color": list(CLASS_COLORS[c])})
    return dets, hm_combined


def _jet(v):
    """Colormap jet sederhana. v: array 0..1 -> (...,3) float 0..1."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def draw_heatmap_overlay(pil_img, hm_combined, alpha=0.8, gamma=0.5):
    base = pil_img.convert("RGB")
    W, H = base.size
    hm_img = Image.fromarray((hm_combined * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    hm_arr = np.asarray(hm_img).astype(np.float32) / 255.0

    # normalisasi ke 0..1 + perkuat dengan gamma (<1 = lebih tebal)
    if hm_arr.max() > 0:
        hm_arr = hm_arr / hm_arr.max()
    hm_arr = np.power(hm_arr, gamma)        # ← gamma<1 menebalkan area lemah
    color = _jet(hm_arr)
    base_arr = np.asarray(base).astype(np.float32) / 255.0
    a = hm_arr[..., None] * alpha
    out = base_arr * (1 - a) + color * a
    return Image.fromarray((out * 255).astype(np.uint8))


def draw_detections(pil_img, dets):
    """Gambar bbox + label di atas citra. Return PIL.Image baru."""
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    lw = max(2, int(round(min(W, H) / 300)))
    fs = max(13, int(round(min(W, H) / 45)))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", fs)
    except Exception:
        font = ImageFont.load_default()

    for d in dets:
        x1, y1, x2, y2 = d["box"]
        color = CLASS_COLORS[d["cls"]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = f"{d['name']} {d['score']:.2f}"
        # latar label
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0, y1 - th - 4)
        draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=color)
        draw.text((x1 + 3, ly + 2), label, fill=(255, 255, 255), font=font)
    return img
