"""Extract frozen 1,024-dimensional image features for the CGMacros photo files."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ConvNeXtBase, eval_transform, load_image, sha256_file


class Photos(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = eval_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            return self.transform(load_image(self.paths[index]))
        except OSError:
            return None


def valid_images(batch):
    images = [x for x in batch if x is not None]
    return None if not images else torch.stack(images), np.array([x is not None for x in batch])


def encode(paths, checkpoint, output, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    model = ConvNeXtBase(state["head.weight"].shape[0], imagenet_init=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    result = np.zeros((len(paths), 1024), dtype=np.float32)
    present = np.zeros(len(paths), dtype=bool)
    start = 0
    with torch.inference_mode():
        for images, mask in DataLoader(Photos(paths), batch_size=32, num_workers=6,
                                       collate_fn=valid_images):
            positions = start + np.flatnonzero(mask)
            if len(positions):
                result[positions] = model.embed(images.to(device)).float().cpu().numpy()
                present[positions] = True
            start += len(mask)
    digests = [sha256_file(paths[i]) for i in np.flatnonzero(present)]
    np.savez_compressed(output, embedding=result[present], sha256=np.asarray(digests))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cgmacros", type=Path, required=True)
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extensions = {".jpg", ".jpeg", ".png"}
    paths = sorted(path for path in args.cgmacros.rglob("*") if path.is_file()
                   and any(p.casefold() == "photos" for p in path.parts)
                   and path.suffix.casefold() in extensions)
    if not paths:
        parser.error("No CGMacros photos found")
    for name in ("food101", "nutrition5k"):
        encode(paths, args.encoder_root / name / "best.pt", args.encoder_root / f"{name}-features.npz",
               args.device)


if __name__ == "__main__":
    main()
