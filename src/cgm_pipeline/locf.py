"""Parameter-free LOCF on the pinned loader's raw, common-origin windows."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from cgm_pipeline.training_data import CONTEXTS, HORIZONS, FrozenTSFMLoader


@dataclass(frozen=True)
class LOCFBatch:
    predictions: np.ndarray
    metadata: pd.DataFrame
    unit: str = "mg/dL"


def predict_locf(context: np.ndarray, *, horizon_minutes: int) -> np.ndarray:
    """Repeat each row's endpoint; reject missing inputs rather than filling them.

    The context is (origins, context points), including the forecast origin.
    No target, future value, fitted scaling, or metadata enters this function.
    """
    if horizon_minutes not in HORIZONS:
        raise ValueError("horizon must be 30, 60, or 120 minutes")
    values = np.asarray(context, dtype=float)
    if values.ndim != 2 or values.shape[1] not in CONTEXTS.values():
        raise ValueError("context must have shape (origins, 48/144/288)")
    if not np.isfinite(values).all():
        raise ValueError("LOCF requires finite context values, including the origin")
    return np.repeat(values[:, -1:], HORIZONS[horizon_minutes], axis=1)


def locf_windows(selection: FrozenTSFMLoader, *, context_hours: int,
                 horizon_minutes: int, batch_size: int = 256,
                 max_origins: int | None = None):
    """Forecast one isolated selection, preserving loader order and audit fields.

    ``max_origins`` bounds a deterministic prefix for smoke checks only. With
    the default None, every saved common origin in the selection is retained.
    """
    if context_hours not in CONTEXTS or horizon_minutes not in HORIZONS:
        raise ValueError("task must be one of the nine pinned context/horizon pairs")
    if not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if max_origins is not None and (not isinstance(max_origins, Integral) or max_origins < 1):
        raise ValueError("max_origins must be a positive integer or None")
    remaining = max_origins
    for batch in selection.windows(context_hours=context_hours,
                                   horizon_minutes=horizon_minutes,
                                   normalization="raw", batch_size=batch_size):
        if (batch.unit != "mg/dL" or
                not batch.metadata.normalization.eq("raw").all()):
            raise ValueError("LOCF requires raw mg/dL loader windows")
        if batch.context.shape != (len(batch.metadata), CONTEXTS[context_hours]):
            raise ValueError("context rows or geometry disagree with metadata/task")
        size = len(batch.metadata) if remaining is None else min(remaining, len(batch.metadata))
        metadata = batch.metadata.iloc[:size].copy(deep=True).reset_index(drop=True)
        metadata["model"] = "LOCF"
        metadata["context_hours"] = context_hours
        metadata["horizon_minutes"] = horizon_minutes
        yield LOCFBatch(predict_locf(batch.context[:size], horizon_minutes=horizon_minutes),
                        metadata)
        if remaining is not None:
            remaining -= size
            if remaining == 0:
                return
