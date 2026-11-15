"""Read the pinned, model-agnostic TSFM windows without changing the freeze.

Only glucose arrays are model inputs. Metadata travels separately with each batch.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


class FreezeAuditError(ValueError):
    """A saved artifact disagrees with its pinned schema or scientific role."""


COHORTS = {
    "T1D": {"development": ("Anderson_2016", "Brown_2019", "Lynch_2022"),
            "external": ("Anderson_2016", "Brown_2019", "Lynch_2022")},
    "Prediabetes & T2D": {"development": ("AI_READI_2025",),
                           "external": ("Broll_2021", "Hall_2018", "CGMacros_2025")},
}
PARTITIONS = ("train", "validation", "internal_test", "external_test")
CONTEXTS = {4: 48, 12: 144, 24: 288}
HORIZONS = {30: 6, 60: 12, 120: 24}
FOLDS = {"A": ("Anderson_2016", ("Brown_2019", "Lynch_2022")),
         "B": ("Brown_2019", ("Anderson_2016", "Lynch_2022")),
         "C": ("Lynch_2022", ("Anderson_2016", "Brown_2019"))}
TARGET_PROVENANCE = {"CGMacros_2025": "source_processed_five_minute_mean"}


@dataclass(frozen=True)
class WindowBatch:
    context: np.ndarray
    target: np.ndarray
    metadata: pd.DataFrame
    unit: str = "mg/dL"


class FrozenTSFMLoader:
    """Select a single dataset/partition in a declared cohort and optional T1D fold.

    ``windows`` yields bounded batches. Each call reuses exactly the same saved
    origin IDs; context and horizon arguments only slice the arrays.
    The default manifest is benchmark_inputs/data_freeze.yaml under work_root.
    A custom freeze_path uses the fold configuration next to that manifest.
    """

    def __init__(self, work_root: Path, *, cohort: str, dataset: str,
                 partition: str, fold: str | None = None,
                 freeze_path: Path | None = None):
        if cohort not in COHORTS or partition not in PARTITIONS:
            raise FreezeAuditError("cohort and partition must be explicit allowlist values")
        role = "external" if partition == "external_test" else "development"
        if dataset not in COHORTS[cohort][role]:
            raise FreezeAuditError("dataset is prohibited for this cohort and partition")
        if cohort == "T1D":
            if fold not in FOLDS:
                raise FreezeAuditError("T1D requires an explicit pinned LODO fold")
            heldout, development = FOLDS[fold]
            if (role == "external" and dataset != heldout) or (role == "development" and dataset not in development):
                raise FreezeAuditError("dataset violates T1D fold isolation")
        elif fold is not None:
            raise FreezeAuditError("fold is only valid for T1D")
        self.work_root = Path(work_root)
        self.cohort, self.dataset, self.partition, self.fold = cohort, dataset, partition, fold
        self._freeze_path = (Path(freeze_path) if freeze_path is not None else
                             self.work_root / "benchmark_inputs" / "data_freeze.yaml")
        freeze = yaml.safe_load(self._freeze_path.read_text())
        if freeze.get("specification_version") != "1.5":
            raise FreezeAuditError("unexpected freeze specification version")
        fold_config = yaml.safe_load((self._freeze_path.parent / "t1d_folds.yaml").read_text())
        configured_folds = {name: (value["external"], tuple(value["development"]))
                            for name, value in fold_config["folds"].items()}
        if configured_folds != FOLDS:
            raise FreezeAuditError("pinned T1D fold configuration conflicts with allowlist")
        runs = freeze["runs"]
        base = self.work_root / "build"
        source = base / dataset / runs[dataset]
        if cohort == "T1D":
            fold_dir = base / "T1D_LODO" / runs["T1D_LODO"] / fold
            if role == "development":
                self.origins_path = fold_dir / "development_origins.parquet"
                self.analysis_path = fold_dir / "development_analysis.parquet"
            else:
                self.origins_path = fold_dir / "external_origins.parquet"
                self.analysis_path = source / "cgm_analysis.parquet"
            norm_path = fold_dir / "normalization_stats.json"
            qc_path = fold_dir / "qc_report.json"
        else:
            if dataset in ("AI_READI_2025", "Hall_2018", "CGMacros_2025"):
                source /= "primary"
            self.origins_path = source / "forecast_origins.parquet"
            self.analysis_path = source / "cgm_analysis.parquet"
            norm_path = source / "normalization_stats.json" if role == "development" else None
            qc_path = source / "qc_report.json"
        self._qc = json.loads(qc_path.read_text())
        self._normalization = json.loads(norm_path.read_text()) if norm_path else None
        if self._normalization is not None:
            self._check_normalization(self._normalization)
        self._origins = self._read_origins()

    def _check_normalization(self, stats: dict) -> None:
        if (stats.get("source_partition") != "train" or
                stats.get("interpolated_values_included") is not False or
                stats.get("ddof") != 0 or
                int(stats.get("observed_training_positions", 0)) < 2 or
                not np.isfinite(stats.get("mean_mgdl", np.nan)) or
                not np.isfinite(stats.get("std_mgdl", np.nan)) or stats["std_mgdl"] <= 0):
            raise FreezeAuditError("normalization provenance is not observed training only")
        if self.cohort == "T1D":
            heldout, development = FOLDS[self.fold]
            if (stats.get("heldout_dataset") != heldout or
                    set(stats.get("source_datasets", [])) != set(development)):
                raise FreezeAuditError("fold normalization source datasets are wrong")
        elif self.dataset != "AI_READI_2025" or stats.get("is_observed_required") is not True:
            raise FreezeAuditError("AI-READI training normalization provenance is wrong")

    def _read_origins(self) -> pd.DataFrame:
        required = {"origin_id", "dataset", "participant_id", "natural_segment_id",
                    "partition", "external_dataset", "context_positions", "target_positions",
                    "primary_cohort"}
        schema = pq.read_schema(self.origins_path)
        if not required.issubset(schema.names):
            raise FreezeAuditError("origin artifact lacks required columns")
        frame = pq.read_table(self.origins_path).to_pandas()
        allowed_datasets = (set(FOLDS[self.fold][1]) if self.cohort == "T1D" and
                            self.partition != "external_test" else {self.dataset})
        allowed_partitions = (set(PARTITIONS[:3]) if self.partition != "external_test" else
                              {"external_test"})
        if (not set(frame.dataset).issubset(allowed_datasets) or
                not set(frame.partition).issubset(allowed_partitions)):
            raise FreezeAuditError("origin artifact contains an unexpected dataset or partition")
        frame = frame.loc[frame.dataset.eq(self.dataset) & frame.partition.eq(self.partition)].copy()
        if frame.empty or frame.origin_id.isna().any() or frame.origin_id.duplicated().any():
            raise FreezeAuditError("origin IDs are empty, null, or duplicated")
        if not frame.context_positions.eq(288).all() or not frame.target_positions.eq(24).all():
            raise FreezeAuditError("pinned origins do not have 288/24 geometry")
        if not frame.primary_cohort.eq(self.cohort).all():
            raise FreezeAuditError("origin disease cohort conflicts with selection")
        if self.partition == "external_test":
            if not frame.external_dataset.eq(self.dataset).all():
                raise FreezeAuditError("external-test role conflicts with dataset")
        elif frame.external_dataset.notna().any():
            raise FreezeAuditError("external dataset entered development")
        if self.dataset == "CGMacros_2025" and ("causal_input_model_checked" not in frame or
                                                not frame.causal_input_model_checked.eq(True).all()):
            raise FreezeAuditError("CGMacros causal-input provenance is missing")
        time_col = "timestamp_utc" if self.dataset == "AI_READI_2025" else "timestamp_local"
        if time_col not in frame or frame[time_col].isna().any():
            raise FreezeAuditError("origin time column missing")
        frame = frame.sort_values(["dataset", "participant_id", "natural_segment_id", time_col,
                                   "origin_id"], kind="stable").reset_index(drop=True)
        expected = self._expected_counts()
        observed = (len(frame), frame.participant_id.nunique())
        if observed != expected:
            raise FreezeAuditError(f"frozen manifest count mismatch for {self.dataset}/{self.partition}: "
                                   f"loaded {observed}, expected {expected}")
        return frame

    def _expected_counts(self) -> tuple[int, int]:
        qc = self._qc
        if self.cohort == "T1D":
            if self.partition == "external_test":
                return qc["external_origins"], qc["external_participants_with_origins"]
            key = f"{self.dataset}:{self.partition}"
            return (qc["development_origin_counts_by_dataset_partition"][key],
                    qc["development_participants_with_origins_by_dataset_partition"][key])
        if self.partition == "external_test":
            return (qc["origin_eligibility"]["eligible_origins"],
                    qc["externally_evaluable_participants"])
        return (qc["origin_counts_by_partition"][self.partition],
                qc["origin_participants_by_partition"][self.partition])

    @property
    def counts(self) -> dict[str, int]:
        return {"origins": len(self._origins), "participants": self._origins.participant_id.nunique()}

    @property
    def origin_ids(self) -> tuple[str, ...]:
        return tuple(self._origins.origin_id)

    @property
    def normalization(self) -> dict | None:
        return dict(self._normalization) if self._normalization else None

    def windows(self, *, context_hours: int, horizon_minutes: int,
                normalization: str = "raw", batch_size: int = 256):
        if context_hours not in CONTEXTS or horizon_minutes not in HORIZONS:
            raise ValueError("task must be one of the nine pinned context/horizon pairs")
        if normalization not in ("raw", "training_global"):
            raise ValueError("normalization must be raw or training_global")
        if normalization == "training_global" and self._normalization is None:
            # External selections obtain the same cohort/fold training stats from
            # their development artifact, never from the external analysis file.
            dev = ("Brown_2019" if self.cohort == "T1D" and self.fold == "A" else
                   "Anderson_2016" if self.cohort == "T1D" else "AI_READI_2025")
            donor = FrozenTSFMLoader(self.work_root, cohort=self.cohort, dataset=dev,
                                     partition="train", fold=self.fold, freeze_path=self._freeze_path)
            stats = donor.normalization
        else:
            stats = self._normalization
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        yield from self._stream(CONTEXTS[context_hours], HORIZONS[horizon_minutes], stats if normalization == "training_global" else None, batch_size)

    def _stream(self, context_len: int, target_len: int, stats: dict | None, batch_size: int):
        time_col = "timestamp_utc" if self.dataset == "AI_READI_2025" else "timestamp_local"
        columns = ["dataset", "participant_id", "natural_segment_id", "partition", time_col,
                   "y_mgdl", "is_observed", "is_interpolated"]
        schema = pq.read_schema(self.analysis_path)
        if not set(columns).issubset(schema.names):
            raise FreezeAuditError("analysis artifact lacks required columns")
        selected = self._origins.groupby(["dataset", "participant_id", "natural_segment_id", "partition"], sort=False)
        origin_groups = {key: value for key, value in selected}
        completed = set()
        pending = []
        carry = pd.DataFrame()

        def process(group):
            key = tuple(group.loc[group.index[0], ["dataset", "participant_id", "natural_segment_id", "partition"]])
            if key in completed:
                raise FreezeAuditError("analysis group is noncontiguous")
            completed.add(key)
            if key not in origin_groups:
                return []
            origins = origin_groups[key]
            group = group.sort_values(time_col, kind="stable").reset_index(drop=True)
            if group[time_col].duplicated().any() or not group[time_col].diff().dropna().eq(pd.Timedelta(minutes=5)).all():
                raise FreezeAuditError("analysis grid is not a unique five-minute sequence")
            position = pd.Index(group[time_col]).get_indexer(origins[time_col])
            if (position < 287).any() or (position + 24 >= len(group)).any():
                raise FreezeAuditError("saved origin falls outside 288/24 analysis geometry")
            values = group.y_mgdl.to_numpy(dtype=float)
            observed = group.is_observed.to_numpy(dtype=bool)
            interpolated = group.is_interpolated.to_numpy(dtype=bool)
            result = []
            for offset, (row, index) in enumerate(zip(origins.itertuples(index=False), position)):
                meta = origins.iloc[offset]
                full_context = values[index - 287:index + 1]
                full_target = values[index + 1:index + 25]
                if (len(full_context) != 288 or len(full_target) != 24 or
                        not np.isfinite(full_context).all() or not np.isfinite(full_target).all() or
                        not observed[index] or not observed[index + 1:index + 25].all() or
                        interpolated[index + 1:index + 25].any()):
                    raise FreezeAuditError(f"invalid saved window geometry: {row.origin_id}")
                if (group[time_col].iloc[index - 287] != meta[f"context_start_{time_col.split('_')[1]}"] or
                        group[time_col].iloc[index + 24] != meta[f"target_end_{time_col.split('_')[1]}"]):
                    raise FreezeAuditError(f"origin boundary mismatch: {row.origin_id}")
                context = full_context[-context_len:].copy()
                target = full_target[:target_len].copy()
                if stats is not None:
                    context = (context - stats["mean_mgdl"]) / stats["std_mgdl"]
                    target = (target - stats["mean_mgdl"]) / stats["std_mgdl"]
                audit = {"origin_id": row.origin_id, "dataset": self.dataset,
                         "participant_id": row.participant_id, "cohort": self.cohort,
                         "partition": self.partition, "fold": self.fold,
                         "external_test_role": self.dataset if self.partition == "external_test" else None,
                         "external_tier": ("source_processed_target" if self.dataset == "CGMacros_2025"
                                           else "observed_target" if self.partition == "external_test" else None),
                         "target_provenance": TARGET_PROVENANCE.get(self.dataset, "observed_sensor"),
                         "source_disease_label": meta.get("source_disease_label"),
                         "disease_subgroup": meta.get("disease_subgroup"),
                         "normalization": "training_global" if stats is not None else "raw"}
                result.append((context, target, audit))
            return result

        def batches(records):
            pending.extend(records)
            while len(pending) >= batch_size:
                chunk = pending[:batch_size]
                del pending[:batch_size]
                yield WindowBatch(np.stack([x[0] for x in chunk]),
                                  np.stack([x[1] for x in chunk]),
                                  pd.DataFrame([x[2] for x in chunk]),
                                  "z-score" if stats is not None else "mg/dL")

        keys = ["dataset", "participant_id", "natural_segment_id", "partition"]
        for batch in pq.ParquetFile(self.analysis_path).iter_batches(batch_size=65536, columns=columns):
            frame = pd.concat([carry, batch.to_pandas()], ignore_index=True)
            groups = list(frame.groupby(keys, sort=False, dropna=False))
            carry = groups[-1][1]
            for _, group in groups[:-1]:
                yield from batches(process(group))
        if not carry.empty:
            yield from batches(process(carry))
        if completed.isdisjoint(origin_groups) or not set(origin_groups).issubset(completed):
            raise FreezeAuditError("selected origin group missing from analysis artifact")
        if pending:
            yield WindowBatch(np.stack([x[0] for x in pending]),
                              np.stack([x[1] for x in pending]),
                              pd.DataFrame([x[2] for x in pending]),
                              "z-score" if stats is not None else "mg/dL")
