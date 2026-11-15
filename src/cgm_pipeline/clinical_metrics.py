"""Clinical primitives from the benchmark analysis."""

import hashlib
import numpy as np

def clarke_zone_codes(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return standard Clarke Error Grid zones as integer codes A=0 through E=4."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Clarke inputs must be same-shape finite arrays")
    zone = np.full(truth.shape, 1, dtype=np.int8)  # B is the residual zone.
    a = ((truth < 70.0) & (prediction < 70.0)) | (
        (prediction >= 0.8 * truth) & (prediction <= 1.2 * truth))
    e = ((truth >= 180.0) & (prediction <= 70.0)) | (
        (truth <= 70.0) & (prediction >= 180.0))
    c = (((truth >= 70.0) & (truth <= 290.0)) & (prediction >= truth + 110.0)) | (
        ((truth >= 130.0) & (truth <= 180.0)) & (prediction <= (7.0 / 5.0) * truth - 182.0))
    d = ((truth >= 240.0) & (prediction >= 70.0) & (prediction <= 180.0)) | (
        (truth <= 175.0 / 3.0) & (prediction >= 70.0) & (prediction <= 180.0)) | (
        ((truth >= 175.0 / 3.0) & (truth <= 70.0)) & (prediction >= 1.2 * truth))
    zone[d] = 3
    zone[c] = 2
    zone[e] = 4
    zone[a] = 0
    return zone


def _has_three_consecutive(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] < 3:
        raise ValueError("event vectors require at least three target points")
    return np.lib.stride_tricks.sliding_window_view(mask, 3, axis=1).all(axis=2).any(axis=1)


def _dataset_bootstrap_seed(base_seed: int, key: tuple) -> int:
    digest = hashlib.sha256((str(base_seed) + "|" + "|".join(map(str, key))).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)
