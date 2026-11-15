"""Common-origin windows and shared outcome-blind training samples."""

from dataclasses import dataclass

import numpy as np

from cgm_pipeline.elastic_net import TrainingNormalization, TrainingScope
from cgm_pipeline.origin_sampling import sample_origins
from cgm_pipeline.training_data import CONTEXTS, FOLDS, HORIZONS, FrozenTSFMLoader


PRIMARY_EXTERNAL = ("Broll_2021", "Hall_2018", "CGMacros_2025")
MODEL_SEEDS = (20261011, 20261012, 20261013)
SCOPES = {"primary": TrainingScope("Prediabetes & T2D")}
SCOPES.update({f"t1d-{fold}": TrainingScope("T1D", fold) for fold in FOLDS})


@dataclass
class Windows:
    context: np.ndarray
    target: np.ndarray
    rows: list

    def task(self, hours, minutes):
        return self.context[:, -CONTEXTS[hours]:], self.target[:, :HORIZONS[minutes]]


def load_scope(work, scope, *, full_training=False):
    def loader(dataset, partition):
        return FrozenTSFMLoader(
            work, cohort=scope.cohort, dataset=dataset, partition=partition, fold=scope.fold,
        )

    def read_windows(selections, selected_ids=None):
        context, target, rows = [], [], []
        for selection in selections:
            for batch in selection.windows(context_hours=24, horizon_minutes=120):
                if selected_ids is None:
                    take = np.arange(len(batch.metadata))
                else:
                    take = np.flatnonzero(batch.metadata.origin_id.isin(selected_ids))
                if len(take):
                    context.append(batch.context[take])
                    target.append(batch.target[take])
                    rows.extend(batch.metadata.iloc[take].to_dict("records"))
        if selected_ids is not None and {row["origin_id"] for row in rows} != selected_ids:
            raise ValueError("Sampled-origin manifest and window contents differ")
        return Windows(np.concatenate(context), np.concatenate(target), rows)

    training = [loader(dataset, "train") for dataset in scope.development_datasets]
    normalization = TrainingNormalization.from_loader(training[0])
    if any(TrainingNormalization.from_loader(source) != normalization for source in training):
        raise ValueError("Development datasets disagree on fold normalization")

    identities = []
    for source in training:
        rows = source._origins[["origin_id", "dataset", "participant_id", "partition", "natural_segment_id"]]
        identities.extend(dict(row, cohort=scope.cohort, fold=scope.fold, normalization="raw")
                          for row in rows.to_dict("records"))
    sample = None if full_training else sample_origins(identities, scope=scope, fraction=0.05, seed=20261000)
    selected = None if full_training else {row["origin_id"] for row in sample["origin_rows"]}
    train = read_windows(training, selected)
    validation = [read_windows([loader(dataset, "validation")]) for dataset in scope.development_datasets]
    return train, validation, normalization, sample


def test_windows(work, scope, hours, minutes, *, batch_size=256):
    external = PRIMARY_EXTERNAL if scope.fold is None else (FOLDS[scope.fold][0],)
    for partition, datasets in (("internal_test", scope.development_datasets),
                                ("external_test", external)):
        for dataset in datasets:
            loader = FrozenTSFMLoader(
                work, cohort=scope.cohort, dataset=dataset, partition=partition, fold=scope.fold,
            )
            yield dataset, partition, loader.windows(context_hours=hours, horizon_minutes=minutes,
                                                     batch_size=batch_size)
