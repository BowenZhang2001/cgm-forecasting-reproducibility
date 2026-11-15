"""Shared model, transform, metric and provenance helpers."""
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # fixed upstream constants, not fitted
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBED_DIM = 1024


class ConvNeXtBase(nn.Module):
    """TorchVision ConvNeXt-Base; embedding = 1024-d output of the classifier's LayerNorm+Flatten
    (i.e. global-average-pooled features right before the final Linear)."""

    def __init__(self, num_outputs, imagenet_init=True, stochastic_depth_prob=0.2):
        super().__init__()
        w = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if imagenet_init else None
        m = torchvision.models.convnext_base(weights=w, stochastic_depth_prob=stochastic_depth_prob)
        self.features, self.avgpool = m.features, m.avgpool
        self.norm = m.classifier[0]          # LayerNorm2d(1024)
        self.flatten = m.classifier[1]
        self.head = nn.Linear(EMBED_DIM, num_outputs)

    def embed(self, x):
        return self.flatten(self.norm(self.avgpool(self.features(x))))

    def forward(self, x):
        return self.head(self.embed(x))


def load_image(path):
    """Decode with PIL, apply EXIF orientation, convert to RGB."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def train_transform(size=224):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(0.2),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size=224):
    """Deterministic inference transform: bicubic resize short side to 256*size/224, center crop."""
    return T.Compose([
        T.Resize(round(size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def regression_metrics(y_true, y_pred, names):
    out = {}
    for j, n in enumerate(names):
        t, p = y_true[:, j], y_pred[:, j]
        err = p - t
        ss_res = float((err ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        out[n] = dict(
            n=int(len(t)),
            mse=float((err ** 2).mean()),
            rmse=float(np.sqrt((err ** 2).mean())),
            mae=float(np.abs(err).mean()),
            mae_pct_of_mean=float(np.abs(err).mean() / max(abs(t.mean()), 1e-9) * 100),
            r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            pearson_r=float(np.corrcoef(t, p)[0, 1]) if t.std() > 0 and p.std() > 0 else float("nan"),
            mean_true=float(t.mean()), mean_pred=float(p.mean()),
        )
    return out


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_lines(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def environment_info():
    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return dict(
        python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
        torchvision=torchvision.__version__, numpy=np.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        git_commit=commit, git_dirty=dirty, argv=sys.argv, host=os.uname().nodename,
    )


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
