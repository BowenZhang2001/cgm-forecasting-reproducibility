"""Nutrition5k dish tables, leakage-safe validation split and view-level datasets."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common import load_image
from src.n5k.metadata import TARGETS, read_dish_metadata, read_ids

# Physically implausible / clearly erroneous labels. Applied to train/val dishes before fitting;
# test is reported both in full (official) and with the same rule ("plausible" subset).
MAX_KCAL_PER_G = 9.0
MAX_MASS_G = 3000.0
MAX_KCAL = 3000.0
SESSION_GAP_S = 600  # dish ids are unix seconds; dishes < 10 min apart are one capture session


def implausible(df):
    return (df.calories / df.mass.clip(lower=1e-6) > MAX_KCAL_PER_G) | (df.mass > MAX_MASS_G) | (df.calories > MAX_KCAL)


def sessions(dish_ids):
    """Group dish ids into capture sessions by time gaps (incrementally-built dishes share a session)."""
    t = np.array(sorted(int(d.split("_")[1]) for d in dish_ids))
    sid = np.concatenate([[0], np.cumsum(np.diff(t) > SESSION_GAP_S)])
    return {f"dish_{ti}": int(s) for ti, s in zip(t, sid)}


def list_views(root, dish_id):
    root = Path(root)
    views = []
    ov = root / "overhead" / dish_id / "rgb.png"
    if ov.exists():
        views.append(("overhead", str(ov)))
    sf = root / "side_frames" / dish_id
    if sf.exists():
        views += [("side", str(p)) for p in sorted(sf.glob("camera_*_frame_*.jpg"))]
    return views


def build_dish_table(root, val_frac=0.1, seed=0):
    root = Path(root)
    meta, _ = read_dish_metadata(root / "metadata")
    meta = meta.set_index("dish_id")
    train_ids = read_ids(root / "dish_ids/splits/rgb_train_ids.txt")
    test_ids = read_ids(root / "dish_ids/splits/rgb_test_ids.txt")
    assert not set(train_ids) & set(test_ids)

    sess = sessions(train_ids)
    uniq = np.array(sorted(set(sess.values())))
    rng = np.random.default_rng(seed)
    val_sess = set(rng.choice(uniq, size=round(len(uniq) * val_frac), replace=False).tolist())

    rows = []
    for split, ids in (("train", train_ids), ("test", test_ids)):
        for d in ids:
            r = meta.loc[d]
            sub = split
            if split == "train":
                sub = "val" if sess[d] in val_sess else "fit"
            rows.append(dict(dish_id=d, official_split=split, split=sub,
                             session=sess.get(d, -1), mass=r.mass, **{k: r[k] for k in TARGETS}))
    df = pd.DataFrame(rows)
    df["implausible"] = implausible(df)
    views = {d: list_views(root, d) for d in df.dish_id}
    df["n_overhead"] = [sum(v[0] == "overhead" for v in views[d]) for d in df.dish_id]
    df["n_side"] = [sum(v[0] == "side" for v in views[d]) for d in df.dish_id]
    return df, views


class TargetScaler:
    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    @classmethod
    def fit(cls, y):
        return cls(y.mean(0), y.std(0))

    def transform(self, y):
        return (y - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean

    def to_dict(self):
        return dict(targets=TARGETS, mean=self.mean.tolist(), std=self.std.tolist())


class ViewDataset(Dataset):
    """items: list of (dish_id, view_type, path); targets: dict dish_id -> normalized target vector."""

    def __init__(self, items, targets, tf):
        self.items, self.targets, self.tf = items, targets, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, _, p = self.items[i]
        return self.tf(load_image(p)), torch.from_numpy(self.targets[d]), i


def sample_epoch_items(dish_ids, views, per_dish, rng):
    """Each dish contributes `per_dish` random views per epoch (dish-balanced). The view type
    (overhead / side) is drawn uniformly among the types the dish has, so the single overhead image
    is not swamped by up to 20 side frames."""
    items = []
    for d in dish_ids:
        by_type = {}
        for v in views[d]:
            by_type.setdefault(v[0], []).append(v)
        types = sorted(by_type)
        for _ in range(per_dish):
            lst = by_type[types[rng.integers(len(types))]]
            items.append((d, *lst[rng.integers(len(lst))]))
    return items


def all_view_items(dish_ids, views):
    return [(d, *v) for d in dish_ids for v in views[d]]
