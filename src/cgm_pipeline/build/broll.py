"""Broll external-test adapter and auditable build orchestration."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from cgm_pipeline.build.common import (
    Rules, align_and_aggregate, apply_boundary_washout, assert_cleaning_qc,
    common_origins, interpolate_short_gaps, make_natural_segments, standardize_glucose,
)
from cgm_pipeline.ingestion.broll import (
    DATASET, REPO_ROOT, checked_file, resolve_work_root, sha256_file, write_json_new,
)

INGESTION_COLUMNS = (
    "source_row_number", "source_participant_id", "analysis_id",
    "source_time_unix_seconds", "source_time_fixed_EST", "source_gl",
)
COHORT = "Prediabetes & T2D"
SUBGROUP = "T2D"
PARTITION = "external_test"


def read_verified_ingestion(work_root: Path, run_id: str, config: dict) -> tuple[pd.DataFrame, dict, Path]:
    if not run_id or "/" in run_id or ".." in run_id:
        raise ValueError("unsafe ingestion run ID")
    run_dir = work_root / "ingestion" / DATASET / run_id
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id or manifest.get("dataset") != DATASET:
        raise ValueError("ingestion manifest dataset or run ID mismatch")
    if (manifest["source_files"]["embedded_rda"]["sha256"] != config["rda_sha256"] or
            manifest["source_files"]["package_tarball"]["sha256"] != config["tarball_sha256"]):
        raise ValueError("ingestion source hashes differ from Broll configuration")
    files = {record["file"]: record for record in manifest["output_files"]}
    record = files.get("source_ingestion.csv")
    if record is None:
        raise ValueError("ingestion table missing from manifest")
    path = run_dir / "source_ingestion.csv"
    checked_file(path, record["sha256"], int(record["size_bytes"]))
    table = pd.read_csv(path, dtype={"source_participant_id": "string", "analysis_id": "string",
                                     "source_time_fixed_EST": "string", "source_gl": "string"})
    if tuple(table.columns) != INGESTION_COLUMNS or table.empty:
        raise ValueError("unexpected source-ingestion table schema")
    if not np.array_equal(table["source_row_number"].to_numpy(), np.arange(1, len(table) + 1)):
        raise ValueError("source row order changed")
    if table[list(INGESTION_COLUMNS)].isna().any().any():
        raise ValueError("unexpected missing Broll source value")
    if table["source_participant_id"].nunique() != 5 or len(table) != 13866:
        raise ValueError("Broll ingestion differs from verified acquisition inventory")
    return table, manifest, run_dir


def source_clock(table: pd.DataFrame) -> pd.Series:
    """Verify fixed EST evidence, then return the unchanged naive local clock."""
    parsed = pd.to_datetime(table["source_time_fixed_EST"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    seconds = pd.to_numeric(table["source_time_unix_seconds"], errors="raise").astype("int64")
    fixed = timezone(timedelta(hours=-5))
    from_epoch = pd.to_datetime(seconds, unit="s", utc=True).dt.tz_convert(fixed).dt.tz_localize(None)
    if not parsed.reset_index(drop=True).equals(from_epoch.reset_index(drop=True)):
        raise ValueError("Broll fixed-EST source clock does not match epoch evidence")
    return parsed


def validate_broll_evidence(config: dict, ingestion_manifest: dict) -> None:
    source = ingestion_manifest["source_metadata"]
    if (config["source_timestamp_policy"] != "preserve_source_encoded_fixed_EST_without_DST_adjustment" or
            source["timestamp_tzone_attribute"] != "EST" or
            config["original_glucose_unit"] != "mg/dL" or source["original_glucose_unit"] != "mg/dL" or
            config["source_disease_label"] != "Type 2 diabetes" or
            config["source_disease_label_scope"] != "dataset_and_study_level" or
            source["source_disease_label_field"] is not None or
            config["adult_eligibility_status"] != "confirmed_at_parent_study_level_via_documented_linkage" or
            config["parent_study_eligible_age_min_years"] < 18 or
            source["individual_ages"] is not None):
        raise ValueError("Broll source evidence does not support the prespecified external cohort")


def build_frames(table: pd.DataFrame, config: dict, ingestion_manifest: dict,
                 rules: Rules) -> dict[str, object]:
    validate_broll_evidence(config, ingestion_manifest)
    source_ids = table[["analysis_id", "source_participant_id"]].drop_duplicates()
    if source_ids["analysis_id"].duplicated().any() or source_ids["source_participant_id"].duplicated().any():
        raise ValueError("Broll participant ID mapping is not one-to-one")
    clocks = source_clock(table)
    values, glucose_qc = standardize_glucose(table["source_gl"], config["original_glucose_unit"], rules)
    if values.isna().any():
        raise ValueError("Broll has an unexpected nonnumeric source glucose value")
    source_rows = pd.DataFrame({
        "dataset": DATASET, "participant_id": table["analysis_id"].astype(str),
        "timestamp_local": clocks, "y_mgdl": values,
    })
    observed, alignment_qc = align_and_aggregate(source_rows, rules)
    seconds_after_grid = ((clocks.dt.hour * 3600 + clocks.dt.minute * 60 + clocks.dt.second)
                          % (rules.grid_minutes * 60))
    alignment_qc["exact_half_grid_ties"] = int((seconds_after_grid == rules.grid_minutes * 30).sum())
    grid, segments = make_natural_segments(observed, rules)
    partitioned, washout_qc = apply_boundary_washout(grid, rules)
    partitioned["partition"] = PARTITION
    partitioned["external_dataset"] = DATASET
    analysis, interpolation_qc = interpolate_short_gaps(partitioned, rules)
    origins, origin_qc = common_origins(analysis, rules)
    metadata = {
        "primary_cohort": COHORT, "source_disease_label": config["source_disease_label"],
        "source_disease_label_scope": config["source_disease_label_scope"],
        "disease_subgroup": SUBGROUP, "device": config["device"],
        "utc_offset_minutes": -300,
    }
    for frame in (observed, partitioned, analysis):
        for key, value in metadata.items():
            frame[key] = value
        frame["source_participant_id"] = frame["participant_id"].map(
            dict(zip(source_ids["analysis_id"], source_ids["source_participant_id"])))
    observed["partition"] = PARTITION
    observed["external_dataset"] = DATASET
    for key, value in metadata.items():
        origins[key] = value
    assert_cleaning_qc(observed, analysis, origins, rules)
    return {
        "source_ids": source_ids, "observed": observed, "grid": grid,
        "segments": segments, "partitioned": partitioned, "analysis": analysis,
        "origins": origins, "glucose_qc": glucose_qc, "alignment_qc": alignment_qc,
        "washout_qc": washout_qc, "interpolation_qc": interpolation_qc, "origin_qc": origin_qc,
    }


def build_reports(frames: dict[str, object], table: pd.DataFrame, config: dict,
                  rules: Rules) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    source_ids = frames["source_ids"]
    observed = frames["observed"]
    grid = frames["grid"]
    segments = frames["segments"]
    partitioned = frames["partitioned"]
    analysis = frames["analysis"]
    origins = frames["origins"]
    source_count = len(table)
    participant_count = len(source_ids)
    observed_count = len(observed)
    grid_count = len(grid)
    final_grid_count = len(analysis)
    observed_after_washout = int(analysis["is_observed"].sum())
    participants_after_washout = analysis["participant_id"].nunique()
    evaluable = origins["participant_id"].nunique()
    flow = [
        ("source_ingestion", "source_rows", source_count, participant_count, "verified unchanged source"),
        ("fixed_EST_local_clock", "source_rows", source_count, participant_count, "no DST or UTC reinterpretation"),
        ("glucose_standardization", "source_rows", source_count, participant_count, "mg/dL; conversion and clipping counted"),
        ("five_minute_alignment", "observed_grid_positions", observed_count, participant_count, "duplicate median aggregation"),
        ("adult_eligibility", "observed_grid_positions", observed_count, participant_count, "parent-study adult criterion"),
        ("disease_label_eligibility", "observed_grid_positions", observed_count, participant_count, "source-level Type 2 diabetes"),
        ("treatment_period", "observed_grid_positions", observed_count, participant_count, "not applicable to Broll"),
        ("prespecified_exclusions", "observed_grid_positions", observed_count, participant_count, "none for Broll"),
        ("adverse_event_deletion", "observed_grid_positions", observed_count, participant_count, "no source adverse-event information"),
        ("unparseable_and_nonnumeric", "observed_grid_positions", observed_count, participant_count, "zero source rows"),
        ("natural_grid", "grid_positions", grid_count, participant_count, "25+ missing positions separate segments"),
        ("boundary_washout", "grid_positions", final_grid_count, participants_after_washout, "first 24 positions per segment removed"),
        ("external_partition", "grid_positions", final_grid_count, participants_after_washout, "whole dataset external; no development split"),
        ("short_gap_interpolation", "grid_positions", final_grid_count, participants_after_washout, "bounded gaps up to 24 positions"),
        ("common_origins", "origins", len(origins), evaluable, "288 context; 24 observed target; 30-minute stride"),
    ]
    qc_summary = pd.DataFrame(flow, columns=["stage", "row_unit", "rows_after", "participants_after", "note"])
    qc_summary["rows_before"] = qc_summary["rows_after"].shift(1).fillna(source_count).astype("int64")
    qc_summary["row_unit_before"] = qc_summary["row_unit"].shift(1).fillna("source_rows")
    qc_summary["participants_before"] = qc_summary["participants_after"].shift(1).fillna(participant_count).astype("int64")
    cohort = source_ids.rename(columns={"analysis_id": "participant_id"}).sort_values("participant_id").copy()
    cohort["dataset"] = DATASET
    cohort["primary_cohort"] = COHORT
    cohort["disease_subgroup"] = SUBGROUP
    cohort["source_disease_label"] = config["source_disease_label"]
    cohort["source_disease_label_scope"] = config["source_disease_label_scope"]
    cohort["adult_eligibility"] = config["adult_eligibility_status"]
    cohort["individual_age_years"] = pd.NA
    cohort["partition"] = PARTITION
    cohort["external_dataset"] = DATASET
    origin_counts = origins.groupby("participant_id").size().to_dict()
    cohort["common_origin_count"] = cohort["participant_id"].map(origin_counts).fillna(0).astype("int64")
    cohort["externally_evaluable"] = cohort["common_origin_count"] > 0
    exclusions = []
    for _, row in cohort.loc[~cohort["externally_evaluable"]].iterrows():
        exclusions.append({"dataset": DATASET, "participant_id": row["participant_id"],
                           "stage": "common_origins", "reason": "insufficient_eligible_history_or_observed_target",
                           "removed_source_rows": 0, "excluded_from_evaluation": True})
    exclusion_log = pd.DataFrame(exclusions, columns=["dataset", "participant_id", "stage", "reason",
                                                    "removed_source_rows", "excluded_from_evaluation"])
    participant_rows = []
    for _, row in cohort.iterrows():
        pid = row["participant_id"]
        part = analysis.loc[analysis["participant_id"] == pid]
        n = len(part)
        observed_n = int(part["is_observed"].sum())
        interp_n = int(part["is_interpolated"].sum())
        participant_rows.append({
            "dataset": DATASET, "participant_id": pid,
            "source_rows": int((table["analysis_id"] == pid).sum()),
            "aligned_observed_positions": int((observed["participant_id"] == pid).sum()),
            "natural_segments": int((segments["participant_id"] == pid).sum()),
            "post_washout_grid_positions": n,
            "post_washout_observed_positions": observed_n,
            "observed_duration_minutes": observed_n * rules.grid_minutes,
            "observed_proportion": observed_n / n if n else None,
            "interpolated_positions": interp_n,
            "interpolated_proportion": interp_n / n if n else None,
            "common_origins": origin_counts.get(pid, 0),
        })
    participant_qc = pd.DataFrame(participant_rows)
    qc = {
        "status": "PASS", "dataset": DATASET, "role": "whole_dataset_external_test",
        "source_rows": source_count, "source_participants": participant_count,
        "glucose_standardization": frames["glucose_qc"],
        "alignment": frames["alignment_qc"],
        "natural_segments": len(segments), "natural_grid_positions": grid_count,
        "natural_grid_missing_positions": grid_count - observed_count,
        "boundary_washout": frames["washout_qc"],
        "observed_positions_after_washout": observed_after_washout,
        "interpolation": frames["interpolation_qc"],
        "origin_eligibility": frames["origin_qc"],
        "origins_by_source_participant": dict(zip(cohort["source_participant_id"], cohort["common_origin_count"].astype(int))),
        "externally_evaluable_participants": int(evaluable),
        "excluded_from_evaluation_participants": participant_count - int(evaluable),
        "partitions": [PARTITION], "normalization_estimated_from_broll": False,
        "fixed_EST_clock_preserved": True,
        "applicable_stop_conditions": "all passed",
        "not_applicable_stop_conditions": ["development normalization", "cross-cohort overlap", "T1D fold overlap", "fixed exclusions from other datasets"],
    }
    return qc_summary, cohort, exclusion_log, participant_qc, qc


def write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as output:
        frame.to_csv(output, index=False)


def file_record(path: Path) -> dict:
    return {"file": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run(work_root: Path, ingestion_run_id: str) -> Path:
    config_path = REPO_ROOT / "configs/datasets/broll.yaml"
    pipeline_path = REPO_ROOT / "configs/pipeline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    if config["dataset"] != DATASET or pipeline["specification"]["version"] != "1.5":
        raise ValueError("unexpected Broll or pipeline configuration")
    rules = Rules.from_config(pipeline["cleaning"])
    table, ingestion_manifest, ingestion_dir = read_verified_ingestion(work_root, ingestion_run_id, config)
    frames = build_frames(table, config, ingestion_manifest, rules)
    qc_summary, cohort, exclusion_log, participant_qc, qc = build_reports(frames, table, config, rules)
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = work_root / "build" / DATASET / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    source_files = ingestion_manifest["source_files"]
    data_manifest = pd.DataFrame([
        {"dataset": DATASET, "artifact_kind": kind, "file_name": record["path"],
         "size_bytes": record["size_bytes"], "sha256": record["sha256"],
         "source_release": config["package_version"], "device": config["device"],
         "original_glucose_unit": config["original_glucose_unit"],
         "source_timezone_evidence": ingestion_manifest["source_metadata"]["timestamp_tzone_attribute"],
         "source_disease_label": config["source_disease_label"],
         "source_disease_label_scope": config["source_disease_label_scope"],
         "download_date": ingestion_manifest["source_acquisition"]["download_date"],
         "software_license": config["software_license"],
         "data_reuse_license_status": config["data_reuse_license_status"]}
        for kind, record in sorted(source_files.items())
    ])
    write_csv_new(run_dir / "data_manifest.csv", data_manifest)
    write_csv_new(run_dir / "cohort_manifest.csv", cohort)
    write_csv_new(run_dir / "exclusion_log.csv", exclusion_log)
    write_csv_new(run_dir / "qc_summary.csv", qc_summary)
    write_csv_new(run_dir / "participant_qc.csv", participant_qc)
    for name, frame in (
        ("cgm_observed.parquet", frames["observed"]),
        ("cgm_partitioned.parquet", frames["partitioned"]),
        ("cgm_analysis.parquet", frames["analysis"]),
        ("forecast_origins.parquet", frames["origins"]),
    ):
        frame.to_parquet(run_dir / name, engine="pyarrow", index=False)
    write_json_new(run_dir / "normalization_stats.json", {
        "status": "not_applicable_external_only", "estimated_from_broll": False,
        "training_mean_mgdl": None, "training_std_mgdl": None,
    })
    write_json_new(run_dir / "qc_report.json", qc)
    code_paths = ["configs/pipeline.yaml", "configs/datasets/broll.yaml",
                  "src/cgm_pipeline/build/common.py", "src/cgm_pipeline/build/broll.py",
                  "scripts/build/build_broll.py", "src/cgm_pipeline/ingestion/broll.py"]
    manifest = {
        "dataset": DATASET, "run_id": run_id,
        "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_ingestion_run_id": ingestion_run_id,
        "source_ingestion_manifest_sha256": sha256_file(ingestion_dir / "run_manifest.json"),
        "source_ingestion_table_sha256": sha256_file(ingestion_dir / "source_ingestion.csv"),
        "source_files": source_files, "code_commit_at_build": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "code_file_sha256": {name: sha256_file(REPO_ROOT / name) for name in code_paths},
        "dependency_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
        "specification_version": pipeline["specification"]["version"],
        "rules": pipeline["cleaning"], "partition": PARTITION,
        "normalization_estimated_from_broll": False,
        "output_files": [file_record(path) for path in sorted(run_dir.iterdir()) if path.is_file()],
    }
    write_json_new(run_dir / "build_manifest.json", manifest)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingestion-run-id", required=True,
                        help="verified Broll Step 0–1 ingestion run")
    parser.add_argument("--work-root", type=Path,
                        help="defaults to CGM_FORECAST_WORK_ROOT or sibling work")
    args = parser.parse_args()
    print(run(resolve_work_root(args.work_root), args.ingestion_run_id))


if __name__ == "__main__":
    main()
