"""CGMacros Dexcom adapter with verified recovery of genuine five-minute samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from cgm_pipeline.build.common import Rules, standardize_glucose
from cgm_pipeline.build.external import save_external
from cgm_pipeline.ingestion.broll import REPO_ROOT, analysis_id, resolve_work_root, sha256_file

DATASET = "CGMacros_2025"


def recover_dexcom_observations(times: pd.Series, glucose: pd.Series,
                                min_run: int = 11, tolerance: float = 1e-6) -> tuple[np.ndarray, dict]:
    """Keep only device anchors in uniquely validated five-minute interpolation runs.

    The published CSV interpolates device readings to one-minute rows. A long
    contiguous run is usable only when exactly one clock-minute phase reproduces
    every intervening value as linear interpolation, and its anchors are integer
    device values. Short or ambiguous runs are never asserted to be observed.
    """
    stamp = pd.to_datetime(times, format="%Y-%m-%d %H:%M:%S", errors="raise")
    if stamp.isna().any() or stamp.duplicated().any() or not stamp.is_monotonic_increasing:
        raise ValueError("CGMacros timestamp is missing, duplicated, or out of order")
    values = pd.to_numeric(glucose, errors="coerce").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("nonfinite CGMacros Dexcom value")
    good = np.isfinite(values)
    clock_minutes = (stamp.dt.hour * 60 + stamp.dt.minute).to_numpy()
    breaks = (~good) | stamp.diff().ne(pd.Timedelta(minutes=1)).to_numpy()
    groups = np.cumsum(breaks)
    selected: list[np.ndarray] = []
    details = []
    short_rows = 0
    for _, positions in pd.Series(np.flatnonzero(good)).groupby(groups[good]):
        run = positions.to_numpy(dtype=int)
        if len(run) < min_run:
            short_rows += len(run)
            continue
        fits = []
        for phase in range(5):
            anchors = run[clock_minutes[run] % 5 == phase]
            errors = []
            for left, right in zip(anchors[:-1], anchors[1:]):
                if right - left == 5:
                    expected = values[left] + (values[right] - values[left]) * np.arange(1, 5) / 5
                    errors.extend(np.abs(values[left + 1:right] - expected))
            if errors and max(errors) <= tolerance:
                fits.append((phase, anchors, len(errors)))
        if len(fits) != 1:
            raise ValueError("CGMacros original five-minute phase is not uniquely verified")
        phase, anchors, checked = fits[0]
        if np.any(np.abs(values[anchors] - np.rint(values[anchors])) > tolerance):
            raise ValueError("CGMacros recovered device anchor is not integer mg/dL")
        confirmation = None
        prefix_lengths = list(range(min_run, min(len(run), 61), 5))
        prefix_lengths += list(range(60, len(run), 60))
        prefix_lengths.append(len(run))
        for length in sorted(set(prefix_lengths)):
            prefix = run[:length]
            prefix_fits = []
            for candidate_phase in range(5):
                prefix_anchors = prefix[clock_minutes[prefix] % 5 == candidate_phase]
                if np.any(np.abs(values[prefix_anchors] - np.rint(values[prefix_anchors])) > tolerance):
                    continue
                residuals = []
                for left, right in zip(prefix_anchors[:-1], prefix_anchors[1:]):
                    if right - left == 5:
                        expected = values[left] + (values[right] - values[left]) * np.arange(1, 5) / 5
                        residuals.extend(np.abs(values[left + 1:right] - expected))
                if residuals and max(residuals) <= tolerance:
                    prefix_fits.append(candidate_phase)
            if prefix_fits == [phase]:
                confirmation = stamp.iloc[prefix[-1]]
                break
        if confirmation is None:
            raise ValueError("Dexcom phase cannot be confirmed from its own source run")
        selected.append(anchors)
        details.append({"source_start_time": str(stamp.iloc[run[0]]),
                        "source_end_time": str(stamp.iloc[run[-1]]),
                        "phase_confirmed_by_local_time": str(confirmation),
                        "one_minute_rows": len(run), "five_minute_phase": phase,
                        "verified_interpolated_rows": checked,
                        "inferred_native_anchor_rows": len(anchors)})
    if not selected:
        raise ValueError("no verified Dexcom device readings")
    positions = np.concatenate(selected)
    if len(np.unique(positions)) != len(positions):
        raise ValueError("duplicate recovered source row")
    return positions, {"validated_runs": len(details), "short_unverified_rows": short_rows,
                       "published_nonmissing_dexcom_rows": int(good.sum()),
                       "inferred_native_anchor_rows": len(positions), "runs": details}


def native_dependency_times(times: pd.Series, anchors: np.ndarray, detail: dict) -> pd.Series:
    """Latest native reading needed by each verified published minute; unknown stays NaT."""
    stamp = pd.to_datetime(times, format="%Y-%m-%d %H:%M:%S", errors="raise")
    dependency = pd.Series(pd.NaT, index=times.index, dtype="datetime64[ns]")
    for run in detail["runs"]:
        start = pd.Timestamp(run["source_start_time"])
        end = pd.Timestamp(run["source_end_time"])
        run_anchors = anchors[(stamp.iloc[anchors] >= start).to_numpy() &
                              (stamp.iloc[anchors] <= end).to_numpy()]
        if len(run_anchors) < 2:
            raise ValueError("validated source run lacks two anchors")
        positions = np.flatnonzero(stamp.between(stamp.iloc[run_anchors[0]], stamp.iloc[run_anchors[-1]]))
        right = np.searchsorted(run_anchors, positions, side="left")
        if np.any(right >= len(run_anchors)):
            raise ValueError("source dependency extends beyond verified anchors")
        dependency.iloc[positions] = stamp.iloc[run_anchors[right]].to_numpy()
    return dependency


def trailing_five_minute_bins(frame: pd.DataFrame, anchors: np.ndarray, detail: dict,
                              participant_id: str, source_id: str,
                              rules: Rules) -> tuple[pd.DataFrame, pd.DataFrame, dict, int]:
    """Nonoverlapping trailing means; retain every complete bin's five source contributors."""
    timestamp = pd.to_datetime(frame.Timestamp, format="%Y-%m-%d %H:%M:%S", errors="raise")
    values, glucose_qc = standardize_glucose(frame["Dexcom GL"], "mg/dL", rules)
    dependency = native_dependency_times(frame.Timestamp, anchors, detail)
    rows = pd.DataFrame({"participant_id": participant_id, "source_participant_id": source_id,
        "source_timestamp_local": timestamp, "source_row_number": np.arange(1, len(frame) + 1),
        "source_glucose_original": frame["Dexcom GL"].astype("string"),
        "source_glucose_standardized_mgdl": values,
        "source_value_is_inferred_native_anchor": np.isin(np.arange(len(frame)), anchors),
        "latest_inferred_native_dependency_local": dependency,
        "native_dependency_inferred": dependency.notna()})
    rows["timestamp_local"] = pd.NaT
    rows["phase_confirmed_by_local_time"] = pd.NaT
    rows["source_phase_minutes"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    for run in detail["runs"]:
        phase = int(run["five_minute_phase"])
        mask = rows.source_timestamp_local.between(
            pd.Timestamp(run["source_start_time"]), pd.Timestamp(run["source_end_time"]))
        offset = pd.Timedelta(minutes=phase)
        rows.loc[mask, "timestamp_local"] = (rows.loc[mask, "source_timestamp_local"] - offset).dt.ceil("5min") + offset
        rows.loc[mask, "source_phase_minutes"] = phase
        rows.loc[mask, "phase_confirmed_by_local_time"] = pd.Timestamp(run["phase_confirmed_by_local_time"])
    rows = rows.loc[rows.source_glucose_standardized_mgdl.notna() & rows.timestamp_local.notna()].copy()
    keys = ["participant_id", "timestamp_local"]
    grouped = rows.groupby(keys, sort=True)
    bins = grouped.agg(y_mgdl=("source_glucose_standardized_mgdl", "mean"),
        source_row_count=("source_row_number", "size"),
        first_source_time=("source_timestamp_local", "min"),
        last_source_time=("source_timestamp_local", "max"),
        phase_confirmed_by_local_time=("phase_confirmed_by_local_time", "max"),
        native_dependency_inferred=("native_dependency_inferred", "all"),
        latest_inferred_native_dependency_local=("latest_inferred_native_dependency_local", "max")).reset_index()
    nonempty_phase_aligned_bins = len(bins)
    complete = (bins.source_row_count.eq(5) &
        bins.first_source_time.eq(bins.timestamp_local - pd.Timedelta(minutes=4)) &
        bins.last_source_time.eq(bins.timestamp_local))
    bins = bins.loc[complete].drop(columns=["first_source_time", "last_source_time"]).copy()
    bins["dataset"] = DATASET
    bins["source_participant_id"] = source_id
    bins["source_phase_minutes"] = bins.timestamp_local.dt.minute.mod(5)
    bins["source_value_provenance"] = "source_processed_five_minute_mean"
    bins["is_native_observation"] = False
    provenance = rows.merge(bins[keys], on=keys, how="inner", validate="many_to_one")
    if len(provenance) != 5 * len(bins) or bins.duplicated(keys).any():
        raise ValueError("five-minute bin contributor count mismatch")
    return bins, provenance, glucose_qc, nonempty_phase_aligned_bins


def source_inventory(folder: Path, config: dict) -> list[dict]:
    files = sorted(folder.iterdir(), key=lambda item: item.name)
    if len(files) != config["source_object_count_analysis"] or not all(path.is_file() for path in files):
        raise ValueError("CGMacros analysis-source file inventory differs")
    records = [{"file": path.name, "size": path.stat().st_size,
                "sha256": sha256_file(path)} for path in files]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (digest != config["source_inventory_sha256"] or
            sum(record["size"] for record in records) != config["source_total_size_bytes_analysis"] or
            next(record for record in records if record["file"] == "bio.csv")["sha256"] != config["bio_sha256"]):
        raise ValueError("CGMacros source hash or total size mismatch")
    for record in records:
        record["path"] = str(folder / record["file"])
    return records


def group_hba1c(value: float, config: dict) -> tuple[str, str]:
    if not np.isfinite(value):
        return "excluded_missing_hba1c", "missing"
    if value < config["hba1c_healthy_upper_exclusive"]:
        return "healthy_supplementary", "healthy"
    if value < config["hba1c_prediabetes_upper_exclusive"]:
        return "primary", "prediabetes"
    return "primary", "T2D"


def read_source(work_root: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict, pd.DataFrame, pd.DataFrame]:
    if (config.get("dataset") != DATASET or config.get("source_glucose_column") != "Dexcom GL" or
            config.get("original_glucose_unit") != "mg/dL" or
            config.get("prespecified_excluded_participants") != ["024", "025", "037", "040"]):
        raise ValueError("unexpected CGMacros configuration")
    folder = work_root / "acquisition" / DATASET / "source"
    inventory = source_inventory(folder, config)
    bio = pd.read_csv(folder / "bio.csv", dtype={"subject": "string"})
    if (len(bio) != 45 or bio.subject.isna().any() or bio.subject.duplicated().any() or
            not {"subject", "Age", "A1c PDL (Lab)"}.issubset(bio.columns)):
        raise ValueError("CGMacros bio schema or participant count changed")
    source_ids = bio.subject.map(lambda value: f"{int(value):03d}")
    excluded = set(config["prespecified_excluded_participants"])
    if set(source_ids) & excluded:
        raise ValueError("prespecified CGMacros excluded participant reappeared")
    file_ids = {re.fullmatch(r"CGMacros-(\d{3})\.csv", path.name).group(1)
                for path in folder.glob("CGMacros-*.csv")}
    if file_ids != set(source_ids):
        raise ValueError("CGMacros bio and CGM participant IDs differ")
    ages = pd.to_numeric(bio.Age, errors="raise")
    hba1c = pd.to_numeric(bio["A1c PDL (Lab)"], errors="coerce")
    if ages.isna().any() or (ages < 18).any():
        raise ValueError("CGMacros adult eligibility cannot be established")
    assigned = [group_hba1c(value, config) for value in hba1c]
    people = pd.DataFrame({"source_participant_id": source_ids,
                           "participant_id": source_ids.map(lambda value: analysis_id(DATASET, value)),
                           "source_disease_label": [None] * len(bio),
                           "cohort_key": [value[0] for value in assigned],
                           "disease_subgroup": [value[1] for value in assigned],
                           "hba1c_percent": hba1c, "age_years": ages,
                           "age_band": ages.map(lambda age: "18-39" if age < 40 else "40+"),
                           "device": config["device"]})
    if people.cohort_key.eq("excluded_missing_hba1c").any():
        raise ValueError("CGMacros missing HbA1c requires exclusion flow before cohort build")
    source_tables = []
    provenance_tables = []
    recovery = {}
    source_columns = {}
    merged_rows = 0
    nonempty_bin_count = 0
    raw_time_min = None
    raw_time_max = None
    glucose_counts = []
    rules = Rules.from_config(yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())["cleaning"])
    for source_id, participant_id in zip(people.source_participant_id, people.participant_id):
        path = folder / f"CGMacros-{source_id}.csv"
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        if not {"Timestamp", "Dexcom GL"}.issubset(columns) or len(columns) != len(set(columns)):
            raise ValueError("CGMacros participant source schema changed")
        source_columns[source_id] = columns
        fields = [name for name in ["Unnamed: 0", "Timestamp", "Dexcom GL"] if name in columns]
        frame = pd.read_csv(path, usecols=fields)
        merged_rows += len(frame)
        source_times = pd.to_datetime(frame.Timestamp, format="%Y-%m-%d %H:%M:%S", errors="raise")
        raw_time_min = min(raw_time_min, source_times.min()) if raw_time_min is not None else source_times.min()
        raw_time_max = max(raw_time_max, source_times.max()) if raw_time_max is not None else source_times.max()
        positions, detail = recover_dexcom_observations(frame.Timestamp, frame["Dexcom GL"],
            config["source_min_run_minutes_for_phase_recovery"],
            config["source_interpolation_tolerance_mgdl"])
        recovery[source_id] = detail
        bins, provenance, glucose_qc, nonempty_bins = trailing_five_minute_bins(
            frame, positions, detail, participant_id, source_id, rules)
        nonempty_bin_count += nonempty_bins
        source_tables.append(bins)
        provenance["source_file"] = path.name
        provenance_tables.append(provenance)
        glucose_counts.append(glucose_qc)
    source = pd.concat(source_tables, ignore_index=True)
    bin_provenance = pd.concat(provenance_tables, ignore_index=True)
    exclusions = pd.DataFrame({"source_participant_id": config["prespecified_excluded_participants"],
        "stage": "prespecified_data_quality_exclusion", "reason": "prespecified_problematic_participant_absent_from_uploaded_source",
        "source_rows_removed": 0})
    qc = {"source_merged_one_minute_rows": merged_rows, "source_participants": len(people),
          "prespecified_excluded_ids_absent": config["prespecified_excluded_participants"],
          "adult_eligible_participants": len(people),
          "hba1c_group_participants": people.disease_subgroup.value_counts().to_dict(),
          "missing_hba1c_participants": int(hba1c.isna().sum()),
          "complete_source_processed_five_minute_bins": len(source),
          "complete_bin_contributor_rows": len(bin_provenance),
          "nonempty_phase_aligned_bins": nonempty_bin_count,
          "nonempty_bins_with_incomplete_five_minute_set": nonempty_bin_count - len(source),
          "bins_with_inferred_native_dependency": int(source.native_dependency_inferred.sum()),
          "bins_with_unknown_native_dependency": int((~source.native_dependency_inferred).sum()),
          "glucose_standardization": {key: sum(item[key] for item in glucose_counts) for key in glucose_counts[0]},
          "validated_runs": sum(item["validated_runs"] for item in recovery.values()),
          "short_unverified_one_minute_rows": sum(item["short_unverified_rows"] for item in recovery.values()),
          "published_nonmissing_dexcom_rows": sum(item["published_nonmissing_dexcom_rows"] for item in recovery.values()),
          "source_time_min": str(raw_time_min),
          "source_time_max": str(raw_time_max),
          "source_duplicate_participant_time": int(source.duplicated(["participant_id", "timestamp_local"]).sum()),
          "source_clock_policy": config["timestamp_policy"], "utc_offset": None,
          "source_columns_by_participant": source_columns,
          "source_phase_recovery_by_participant": recovery}
    return source, people, inventory, qc, exclusions, bin_provenance


def run(work_root: Path) -> Path:
    config = yaml.safe_load((REPO_ROOT / "configs/datasets/cgmacros.yaml").read_text())
    pipeline = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())
    if pipeline["specification"]["version"] != "1.5":
        raise ValueError("unexpected cleaning specification")
    source, people, inventory, qc, exclusions, bin_provenance = read_source(work_root, config)
    return save_external(work_root, DATASET, source, people, inventory, config,
        Rules.from_config(pipeline["cleaning"]),
        ["configs/datasets/cgmacros.yaml", "configs/pipeline.yaml",
         "src/cgm_pipeline/build/common.py", "src/cgm_pipeline/build/external.py",
         "src/cgm_pipeline/build/cgmacros.py", "scripts/build/build_cgmacros.py"],
        qc, exclusions, source_processed=True, bin_provenance=bin_provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path)
    args = parser.parse_args()
    print(run(resolve_work_root(args.work_root)))


if __name__ == "__main__":
    main()
