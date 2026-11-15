"""AI-READI v3-compatible CGM development build with UTC-axis cleaning."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
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
from cgm_pipeline.ingestion.broll import REPO_ROOT, resolve_work_root, sha256_file, write_json_new

DATASET = "AI_READI_2025"
TIME = "timestamp_utc"
COHORTS = {
    "pre_diabetes_lifestyle_controlled": ("primary", "Prediabetes & T2D", "prediabetes"),
    "oral_medication_and_or_non_insulin_injectable_medication_controlled":
        ("primary", "Prediabetes & T2D", "non_insulin_medication_controlled"),
    "insulin_dependent": ("primary", "Prediabetes & T2D", "insulin_dependent"),
    "healthy": ("healthy_supplementary", "Healthy", "healthy"),
}
SITE_ZONE = {"UW": ("pst", "America/Los_Angeles"),
             "UCSD": ("pst", "America/Los_Angeles"),
             "UAB": ("cst", "America/Chicago")}
PARTITIONS = ("train", "validation", "internal_test")


def source_path(work_root: Path) -> Path:
    return work_root / "acquisition" / DATASET / "source"


def validate_adapter_config(config: dict) -> None:
    expected_labels = {cohort: {label for label, mapping in COHORTS.items() if mapping[0] == cohort}
                       for cohort in ("primary", "healthy_supplementary")}
    actual_labels = {cohort: set(labels) for cohort, labels in config["cohort_source_labels"].items()}
    policy = config["timestamp_policy"]
    actual_sites = {site: (header, policy[header]) for site, header in policy["site_mapping"].items()}
    if (actual_labels != expected_labels or actual_sites != SITE_ZONE or
            policy["source_reading_suffix"] != "Z" or policy["axis"] != "UTC" or
            config["source_device"] != "Dexcom G6" or config["source_glucose_unit"] != "mg/dL" or
            config["source_files"] != {"participants": "participants.tsv", "cgm_manifest": "manifest.tsv",
                                       "cgm_directory": "dexcom_g6/dexcom_g6"}):
        raise ValueError("AI-READI adapter and dataset configuration differ")


def verify_source_hashes(root: Path, files: pd.DataFrame, config: dict) -> str:
    pins = config["verified_source_sha256"]
    for filename, key in (("participants.tsv", "participants_tsv"),
                          ("manifest.tsv", "cgm_manifest_tsv")):
        if sha256_file(root / filename) != pins[key]:
            raise ValueError(f"AI-READI source hash mismatch: {filename}")
    digest = hashlib.sha256()
    for row in files.sort_values("source_file").itertuples(index=False):
        digest.update(f"{row.source_file}\t{row.size_bytes}\t{row.sha256}\n".encode())
    if digest.hexdigest() != pins["cgm_inventory_digest"]:
        raise ValueError("AI-READI CGM source inventory hash mismatch")
    return digest.hexdigest()


def analysis_id(person_id: str) -> str:
    return f"{len(DATASET)}:{DATASET}{len(person_id)}:{person_id}"


def checked_metadata(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = pd.read_csv(root / "participants.tsv", sep="\t", dtype={"person_id": "string"})
    manifest = pd.read_csv(root / "manifest.tsv", sep="\t", dtype={"person_id": "string"})
    expected_p = {"person_id", "clinical_site", "study_group", "age", "wearable_blood_glucose"}
    expected_m = {"person_id", "glucose_filepath", "glucose_level_record_count", "manufacturer", "manufacturer_model_name"}
    if not expected_p.issubset(participants) or not expected_m.issubset(manifest):
        raise ValueError("unexpected AI-READI metadata schema")
    if len(participants) != 2280 or len(manifest) != 2245 or participants.person_id.duplicated().any() or manifest.person_id.duplicated().any():
        raise ValueError("metadata counts or participant IDs differ from v3-compatible inventory")
    if set(participants.study_group) != set(COHORTS) or set(participants.clinical_site) != set(SITE_ZONE):
        raise ValueError("unknown source study group or clinical site")
    if participants.age.isna().any() or (participants.age < 18).any():
        raise ValueError("adult eligibility differs from source inventory")
    if set(manifest.person_id) != set(participants.loc[participants.wearable_blood_glucose, "person_id"]):
        raise ValueError("CGM manifest differs from participant availability flags")
    if set(manifest.manufacturer_model_name) != {"G6"} or set(manifest.manufacturer) != {"Dexcom"}:
        raise ValueError("unexpected CGM device")
    return participants, manifest


def local_fields(instants: pd.Series, zone: str) -> tuple[pd.Series, pd.Series]:
    """Convert UTC instants with regional DST, retaining repeated fall clocks by offset."""
    local = instants.dt.tz_convert(zone)
    naive = local.dt.tz_localize(None)
    offset = ((naive.astype("int64") - instants.dt.tz_localize(None).astype("int64"))
              // (60 * 1_000_000_000)).astype("int16")
    return naive, offset


def read_cgm(root: Path, participants: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    people = participants.set_index("person_id")
    paths = list((root / "dexcom_g6").rglob("*_DEX.json"))
    if len(paths) != len(manifest):
        raise ValueError("CGM file count differs from source manifest")
    discovered = {path.stem.removesuffix("_DEX"): path for path in paths}
    if len(discovered) != len(paths) or set(discovered) != set(manifest.person_id):
        raise ValueError("CGM file IDs differ from source manifest")
    batches = []
    files = []
    anomaly = []
    unit_counts = Counter()
    event_counts = Counter()
    for item in manifest.itertuples(index=False):
        pid = str(item.person_id)
        path = discovered[pid]
        expected_suffix = f"/dexcom_g6/{pid}/{pid}_DEX.json"
        if not str(item.glucose_filepath).endswith(expected_suffix):
            raise ValueError(f"manifest path mismatch for {pid}")
        data = json.loads(path.read_bytes())
        header = data["header"]
        site = people.loc[pid, "clinical_site"]
        expected_header, zone = SITE_ZONE[site]
        if header.get("timezone") != expected_header:
            raise ValueError(f"source header timezone contradicts site for {pid}")
        if (header.get("schema_id") != {"namespace": "omh", "name": "blood-glucose", "version": 3.0}
                or header.get("modality") != "sensed" or set(data["body"]) != {"cgm"}):
            raise ValueError(f"unexpected CGM schema for {pid}")
        if header.get("patient_id") not in (f"AIREADI-{pid}", pid) or header.get("uuid") != f"AIREADI-{pid}":
            raise ValueError(f"CGM header ID contradicts manifest for {pid}")
        if header["patient_id"] == pid:
            anomaly.append({"person_id": pid, "finding": "header_patient_id_missing_AIREADI_prefix"})
        readings = data["body"]["cgm"]
        if len(readings) != int(item.glucose_level_record_count):
            raise ValueError(f"CGM row count differs from manifest for {pid}")
        values = []
        stamps = []
        for reading in readings:
            event = reading.get("event_type")
            event_counts[event] += 1
            if event != "EGV":
                raise ValueError(f"non-sensor CGM event for {pid}: {event}")
            glucose = reading["blood_glucose"]
            unit_counts[glucose.get("unit")] += 1
            if glucose.get("unit") != "mg/dL":
                raise ValueError(f"unexpected glucose unit for {pid}")
            interval = reading["effective_time_frame"]["time_interval"]
            stamp = interval["start_date_time"]
            if not stamp.endswith("Z") or interval["end_date_time"] != stamp:
                raise ValueError(f"unexpected UTC timestamp or interval for {pid}")
            values.append(glucose["value"])
            stamps.append(stamp)
        utc = pd.to_datetime(pd.Series(stamps), format="ISO8601", utc=True, errors="raise")
        local, offset = local_fields(utc, zone)
        batches.append(pd.DataFrame({
            "dataset": DATASET, "participant_id": analysis_id(pid), "source_participant_id": pid,
            TIME: utc, "source_timestamp_utc": stamps, "timestamp_local": local,
            "utc_offset_minutes": offset, "source_timezone_header": expected_header,
            "source_glucose": pd.Series(values, dtype="string"),
        }))
        files.append({"source_file": str(item.glucose_filepath), "local_file": str(path),
                      "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    rows = pd.concat(batches, ignore_index=True)
    if len(rows) != int(manifest.glucose_level_record_count.sum()):
        raise ValueError("total source rows differ from CGM manifest")
    return rows, pd.DataFrame(files), {"events": dict(event_counts), "units": dict(unit_counts),
                                        "source_anomalies": anomaly}


def assign_partitions(grid: pd.DataFrame, rules: Rules) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select 60/80% boundaries from post-washout observed UTC grid points."""
    parts = []
    cuts = []
    for pid, group in grid.groupby("participant_id", sort=True):
        group = group.sort_values(TIME).copy()
        observed = group.loc[group.is_observed, TIME]
        n = len(observed)
        if n == 0:
            group["partition"] = "train"
            cuts.append({"participant_id": pid, "observed_positions": 0, "train_observed": 0,
                         "validation_observed": 0, "internal_test_observed": 0})
        else:
            a, b = int(np.floor(0.6 * n)), int(np.floor(0.8 * n))
            if a == 0 or b == a or b == n:
                raise ValueError(f"insufficient observed points for three chronological partitions: {pid}")
            first_val, first_test = observed.iloc[a], observed.iloc[b]
            group["partition"] = np.where(group[TIME] < first_val, "train",
                                           np.where(group[TIME] < first_test, "validation", "internal_test"))
            counts = group.loc[group.is_observed, "partition"].value_counts()
            if [int(counts.get(x, 0)) for x in PARTITIONS] != [a, b-a, n-b]:
                raise ValueError("observed partition count mismatch")
            cuts.append({"participant_id": pid, "observed_positions": n, "train_observed": a,
                         "validation_observed": b-a, "internal_test_observed": n-b,
                         "first_validation_utc": first_val, "first_internal_test_utc": first_test})
        parts.append(group)
    result = pd.concat(parts, ignore_index=True)
    result["external_dataset"] = pd.NA
    return result, pd.DataFrame(cuts)


def attach_metadata(frame: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    details = people.set_index("participant_id")
    for key in ("source_participant_id", "primary_cohort", "source_disease_label",
                "disease_subgroup", "age", "clinical_site", "source_timezone_header", "zone"):
        frame[key] = frame.participant_id.map(details[key])
    frame["device"] = "Dexcom G6"
    if TIME in frame:
        frame["timestamp_local"] = pd.NaT
        frame["utc_offset_minutes"] = np.int16(0)
        for zone in details.zone.unique():
            mask = frame.zone.eq(zone)
            frame.loc[mask, "timestamp_local"], frame.loc[mask, "utc_offset_minutes"] = local_fields(frame.loc[mask, TIME], zone)
    return frame


def cohort_build(rows: pd.DataFrame, people: pd.DataFrame, rules: Rules) -> dict:
    if people.empty:
        raise ValueError("empty development cohort")
    ids = set(people.participant_id)
    source = rows.loc[rows.participant_id.isin(ids), ["dataset", "participant_id", TIME, "source_glucose"]].copy()
    values, glucose_qc = standardize_glucose(source.source_glucose, "mg/dL", rules)
    source["y_mgdl"] = values
    invalid = source[TIME].isna() | source.y_mgdl.isna()
    excluded_rows = int(invalid.sum())
    source = source.loc[~invalid, ["dataset", "participant_id", TIME, "y_mgdl"]]
    observed, alignment_qc = align_and_aggregate(source, rules, TIME)
    grid, segments = make_natural_segments(observed, rules, TIME)
    washed, washout_qc = apply_boundary_washout(grid, rules, TIME)
    partitioned, cuts = assign_partitions(washed, rules)
    analysis, interpolation_qc = interpolate_short_gaps(partitioned, rules, TIME)
    observed = attach_metadata(observed, people)
    partitioned = attach_metadata(partitioned, people)
    analysis = attach_metadata(analysis, people)
    origins, origin_qc = common_origins(analysis, rules, TIME)
    origins = attach_metadata(origins, people)
    if not origins.empty:
        origins["context_start_local"] = pd.Series(pd.NaT, index=origins.index, dtype="datetime64[ns]")
        origins["target_end_local"] = pd.Series(pd.NaT, index=origins.index, dtype="datetime64[ns]")
        for zone in people.zone.unique():
            mask = origins.zone.eq(zone)
            origins.loc[mask, "context_start_local"] = local_fields(origins.loc[mask, "context_start_utc"], zone)[0]
            origins.loc[mask, "target_end_local"] = local_fields(origins.loc[mask, "target_end_utc"], zone)[0]
    assert_cleaning_qc(observed, analysis, origins, rules, TIME, role="development")
    train = analysis.loc[analysis.partition.eq("train") & analysis.is_observed, "y_mgdl"]
    if len(train) < 2 or train.isna().any():
        raise ValueError("insufficient genuine training observations for normalization")
    normalization = {"mean_mgdl": float(train.mean()), "std_mgdl": float(train.std(ddof=0)),
                     "observed_training_positions": len(train), "ddof": 0,
                     "source_partition": "train", "is_observed_required": True,
                     "interpolated_values_included": False}
    if normalization["std_mgdl"] <= 0:
        raise ValueError("zero training standard deviation")
    if len(train) != int(cuts.train_observed.sum()) or analysis.loc[analysis.partition.eq("train") & analysis.is_observed, "is_interpolated"].any():
        raise ValueError("normalization source includes nontraining or interpolated positions")
    return locals()


def output_record(path: Path) -> dict:
    return {"file": str(path.name), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run(work_root: Path) -> Path:
    root = source_path(work_root)
    config_path = REPO_ROOT / "configs/datasets/ai_readi.yaml"
    config = yaml.safe_load(config_path.read_text())
    pipeline = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())
    if config["dataset"] != DATASET or config["source_release_status"] != "consistent_with_v3.0.0_not_independently_verified":
        raise ValueError("unexpected AI-READI configuration")
    validate_adapter_config(config)
    rules = Rules.from_config(pipeline["cleaning"])
    people_raw, source_manifest = checked_metadata(root)
    source_rows, source_files, ingestion_qc = read_cgm(root, people_raw, source_manifest)
    inventory_digest = verify_source_hashes(root, source_files, config)
    people = people_raw[["person_id", "clinical_site", "study_group", "age"]].copy()
    people["source_participant_id"] = people.person_id.astype(str)
    people["participant_id"] = people.source_participant_id.map(analysis_id)
    people["cohort_key"] = people.study_group.map(lambda x: COHORTS[x][0])
    people["primary_cohort"] = people.study_group.map(lambda x: COHORTS[x][1])
    people["source_disease_label"] = people.study_group
    people["disease_subgroup"] = people.study_group.map(lambda x: COHORTS[x][2])
    people["source_timezone_header"] = people.clinical_site.map(lambda x: SITE_ZONE[x][0])
    people["zone"] = people.clinical_site.map(lambda x: SITE_ZONE[x][1])
    if people.participant_id.duplicated().any() or people.groupby("participant_id").cohort_key.nunique().gt(1).any():
        raise ValueError("participant appears in both AI-READI cohorts")
    source_rows["source_disease_label"] = source_rows.participant_id.map(people.set_index("participant_id").source_disease_label)
    started = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    out = work_root / "build" / DATASET / run_id
    out.mkdir(parents=True, exist_ok=False)
    source_rows.to_parquet(out / "source_ingestion.parquet", index=False)
    source_files.to_csv(out / "source_file_inventory.csv", index=False)
    metadata_files = [{"source_file": name, "local_file": str(root / name),
                       "size_bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
                      for name in ("participants.tsv", "manifest.tsv")]
    data_manifest = pd.concat([pd.DataFrame(metadata_files), source_files], ignore_index=True)
    data_manifest["source_release_status"] = config["source_release_status"]
    data_manifest["device"] = "Dexcom G6"
    data_manifest["original_glucose_unit"] = "mg/dL"
    data_manifest["source_timestamp_evidence"] = "Z_UTC_with_pst_or_cst_site_header"
    data_manifest["source_disease_label_field"] = "study_group"
    data_manifest.to_csv(out / "data_manifest.csv", index=False)
    reports = {}
    for cohort_key in ("primary", "healthy_supplementary"):
        cohort_people = people.loc[people.cohort_key.eq(cohort_key)].copy()
        built = cohort_build(source_rows, cohort_people, rules)
        cohort_dir = out / cohort_key
        cohort_dir.mkdir()
        for name, frame in (("cgm_observed.parquet", built["observed"]),
                            ("cgm_partitioned.parquet", built["partitioned"]),
                            ("cgm_analysis.parquet", built["analysis"]),
                            ("forecast_origins.parquet", built["origins"])):
            frame.to_parquet(cohort_dir / name, index=False)
        built["cuts"].to_csv(cohort_dir / "partition_cutpoints.csv", index=False)
        counts = built["origins"].groupby(["participant_id", "partition"]).size().rename("origins").reset_index()
        cohort_people = cohort_people.merge(counts.pivot(index="participant_id", columns="partition", values="origins"),
                                            on="participant_id", how="left")
        for partition in PARTITIONS:
            if partition not in cohort_people:
                cohort_people[partition] = 0
            cohort_people[partition] = cohort_people[partition].fillna(0).astype(int)
        cohort_people["cgm_available"] = cohort_people.participant_id.isin(set(built["observed"].participant_id))
        cohort_people["any_origin"] = cohort_people[[*PARTITIONS]].sum(axis=1).gt(0)
        cohort_people["internally_evaluable"] = cohort_people.validation.gt(0) & cohort_people.internal_test.gt(0)
        cohort_people["train_only"] = cohort_people.train.gt(0) & (cohort_people.validation.eq(0) | cohort_people.internal_test.eq(0))
        cohort_people.to_csv(cohort_dir / "cohort_manifest.csv", index=False)
        exclusion = cohort_people.loc[~cohort_people.cgm_available, ["participant_id", "source_participant_id"]].copy()
        exclusion["stage"] = "CGM_availability"
        exclusion["reason"] = "no_source_CGM_file"
        no_origin = cohort_people.loc[cohort_people.cgm_available & ~cohort_people.any_origin,
                                      ["participant_id", "source_participant_id"]].copy()
        no_origin["stage"] = "common_origins"
        no_origin["reason"] = "insufficient_eligible_history_or_observed_target"
        exclusion = pd.concat([exclusion, no_origin], ignore_index=True)
        exclusion.to_csv(cohort_dir / "exclusion_log.csv", index=False)
        part_qc = cohort_people[["participant_id", "source_participant_id", "study_group", "age", "clinical_site",
                                 "cgm_available", "internally_evaluable", "train_only", *PARTITIONS]].copy()
        count_sources = {
            "source_rows": built["source"].groupby("participant_id").size(),
            "aligned_observed_positions": built["observed"].groupby("participant_id").size(),
            "natural_segments": built["segments"].groupby("participant_id").size(),
            "post_washout_grid_positions": built["analysis"].groupby("participant_id").size(),
            "post_washout_observed_positions": built["analysis"].loc[built["analysis"].is_observed].groupby("participant_id").size(),
            "interpolated_positions": built["analysis"].loc[built["analysis"].is_interpolated].groupby("participant_id").size(),
        }
        for column, counts_by_id in count_sources.items():
            part_qc[column] = part_qc.participant_id.map(counts_by_id).fillna(0).astype("int64")
        part_qc["observed_duration_minutes"] = part_qc.post_washout_observed_positions * rules.grid_minutes
        denominator = part_qc.post_washout_grid_positions.replace(0, np.nan)
        part_qc["observed_proportion"] = part_qc.post_washout_observed_positions / denominator
        part_qc["interpolated_proportion"] = part_qc.interpolated_positions / denominator
        part_qc["common_origins"] = part_qc[[*PARTITIONS]].sum(axis=1)
        part_qc.to_csv(cohort_dir / "participant_qc.csv", index=False)
        flow = [
            ("source_metadata", "participants", len(cohort_people), len(cohort_people), "source-provided study_group"),
            ("adult_eligibility", "participants", len(cohort_people), len(cohort_people), "all recorded ages >=18"),
            ("disease_label_eligibility", "participants", len(cohort_people), len(cohort_people), "exact source labels"),
            ("CGM_availability", "participants", int(cohort_people.cgm_available.sum()), int(cohort_people.cgm_available.sum()), "missing CGM files recorded as exclusions"),
            ("source_ingestion", "source_rows", len(built["source"])+built["excluded_rows"], int(cohort_people.cgm_available.sum()), "EGV readings only"),
            ("valid_time_and_glucose", "source_rows", len(built["source"]), int(cohort_people.cgm_available.sum()), "invalid rows excluded"),
            ("five_minute_alignment", "observed_grid_positions", len(built["observed"]), int(cohort_people.cgm_available.sum()), "median aggregation"),
            ("natural_grid", "grid_positions", len(built["grid"]), int(cohort_people.cgm_available.sum()), "25+ missing positions start a new segment"),
            ("boundary_washout", "grid_positions", len(built["partitioned"]), int(built["partitioned"].participant_id.nunique()), "first 24 positions per segment removed"),
            ("chronological_partition", "grid_positions", len(built["partitioned"]), int(built["partitioned"].participant_id.nunique()), "60/20/20 from observed grid points"),
            ("short_gap_interpolation", "grid_positions", len(built["analysis"]), int(built["analysis"].participant_id.nunique()), "bounded within segment and partition"),
            ("common_origins", "origins", len(built["origins"]), int(cohort_people.any_origin.sum()), "288 context, 24 observed target, six-position stride"),
        ]
        flow_report = pd.DataFrame(flow, columns=["stage", "row_unit", "rows_after", "participants_after", "note"])
        flow_report["rows_before"] = flow_report.rows_after.shift(1).fillna(flow_report.rows_after.iloc[0]).astype("int64")
        flow_report["participants_before"] = flow_report.participants_after.shift(1).fillna(flow_report.participants_after.iloc[0]).astype("int64")
        flow_report.to_csv(cohort_dir / "qc_summary.csv", index=False)
        write_json_new(cohort_dir / "normalization_stats.json", built["normalization"])
        qc = {"source_participants": len(cohort_people), "source_rows": len(built["source"])+built["excluded_rows"],
              "CGM_available_participants": int(cohort_people.cgm_available.sum()),
              "adult_eligible_participants": len(cohort_people),
              "disease_label_eligible_participants": len(cohort_people),
              "participants_with_any_origin": int(cohort_people.any_origin.sum()),
              "internally_evaluable_participants": int(cohort_people.internally_evaluable.sum()),
              "train_only_participants": int(cohort_people.train_only.sum()),
              "no_CGM_participants": int((~cohort_people.cgm_available).sum()),
              "no_origin_with_CGM_participants": int((cohort_people.cgm_available & ~cohort_people.any_origin).sum()),
              "invalid_source_rows": built["excluded_rows"],
              "glucose_standardization": built["glucose_qc"],
              "alignment": built["alignment_qc"],
              "observed_grid_positions": len(built["observed"]),
              "natural_segments": len(built["segments"]), "natural_grid_positions": len(built["grid"]),
              "washout": built["washout_qc"], "post_washout_grid_positions": len(built["partitioned"]),
              "partition_observed_positions": built["cuts"][["train_observed", "validation_observed", "internal_test_observed"]].sum().to_dict(),
              "interpolation": built["interpolation_qc"], "origins": built["origin_qc"],
              "origin_participants_by_partition": {p: int(cohort_people[p].gt(0).sum()) for p in PARTITIONS},
              "origin_counts_by_partition": built["origins"].partition.value_counts().to_dict(),
              "source_label_participants": cohort_people.study_group.value_counts().to_dict(),
              "normalization": built["normalization"],
              "applicable_scientific_stop_conditions": "all passed", "status": "PASS"}
        write_json_new(cohort_dir / "qc_report.json", qc)
        reports[cohort_key] = qc
    ingestion_qc.update({
        "source_metadata_participants": len(people_raw),
        "CGM_manifest_participants": len(source_manifest),
        "CGM_files": len(source_files), "source_CGM_rows": len(source_rows),
        "source_time_min_utc": source_rows[TIME].min().isoformat(),
        "source_time_max_utc": source_rows[TIME].max().isoformat(),
        "source_glucose_low_tokens": int(source_rows.source_glucose.eq("Low").sum()),
        "source_glucose_high_tokens": int(source_rows.source_glucose.eq("High").sum()),
        "source_missing_counts": {column: int(count) for column, count in source_rows.isna().sum().items()},
        "source_columns_and_types": {column: str(dtype) for column, dtype in source_rows.dtypes.items()},
        "source_duplicate_participant_UTC_times": int(source_rows.duplicated(["participant_id", TIME]).sum()),
        "source_header_site_pairs": {f"{site}:{header}": int(count) for (site, header), count in
                                     source_rows[["participant_id", "source_timezone_header"]]
                                     .drop_duplicates().assign(clinical_site=lambda x: x.participant_id.map(
                                         people.set_index("participant_id").clinical_site))
                                     .groupby(["clinical_site", "source_timezone_header"]).size().items()},
        "source_label_participants": people.study_group.value_counts().to_dict(),
        "source_label_CGM_rows": source_rows.source_disease_label.value_counts().to_dict(),
        "adult_age_min": int(people.age.min()), "adult_age_max": int(people.age.max()),
    })
    write_json_new(out / "source_ingestion_qc.json", ingestion_qc)
    write_json_new(out / "qc_report.json", reports)
    write_json_new(out / "normalization_stats.json", {key: value["normalization"] for key, value in reports.items()})
    code_files = ["configs/datasets/ai_readi.yaml", "configs/pipeline.yaml", "src/cgm_pipeline/build/common.py",
                  "src/cgm_pipeline/build/ai_readi.py", "scripts/build/build_ai_readi.py"]
    outputs = [output_record(path) | {"file": str(path.relative_to(out))} for path in sorted(out.rglob("*")) if path.is_file()]
    write_json_new(out / "build_manifest.json", {
        "dataset": DATASET, "run_id": run_id, "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_release_status": config["source_release_status"],
        "source_evidence": config["source_evidence"],
        "source_files": metadata_files + source_files.to_dict("records"),
        "verified_cgm_inventory_digest": inventory_digest,
        "code_commit_at_build": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "code_file_sha256": {name: sha256_file(REPO_ROOT / name) for name in code_files},
        "dependency_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
        "rules": pipeline["cleaning"], "output_files": outputs,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path)
    args = parser.parse_args()
    print(run(resolve_work_root(args.work_root)))
