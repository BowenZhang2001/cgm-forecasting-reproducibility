"""Audited T1D baseline adapters and common five-minute dataset builds."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
import json
import subprocess

import numpy as np
import pandas as pd
import yaml

from cgm_pipeline.build.common import (
    DATA_BUILD_COMMAND, Rules, align_and_aggregate, apply_boundary_washout, assert_cleaning_qc,
    common_origins, interpolate_short_gaps, make_natural_segments, standardize_glucose,
)
from cgm_pipeline.ingestion.broll import REPO_ROOT, analysis_id, checked_file, resolve_work_root, sha256_file, write_json_new

DATASETS = ("Anderson_2016", "Brown_2019", "Lynch_2022")
SOURCE_COLUMNS = {
    "Anderson_2016": ["DeidentID", "RecID", "InternalTime", "CGM", "DisplayTime", "DisplayTimeAdjusted"],
    "Brown_2019": ["PtID", "RecID", "DexcomIndex", "DataDtTm", "EventSubType", "CGM", "TransmitterTime", "DataDtTm_adj"],
    "Lynch_2022": ["RecID", "PtID", "DeviceDtTm", "InternalDtTm", "RecordType", "Value", "Units", "SortOrd"],
}


def config_for(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"unknown T1D dataset: {dataset}")
    path = REPO_ROOT / "configs/datasets" / (dataset.split("_")[0].lower() + ".yaml")
    config = yaml.safe_load(path.read_text())
    if config["dataset"] != dataset or config["original_glucose_unit"] != "mg/dL":
        raise ValueError("unexpected T1D configuration")
    return config


def source_files(work_root: Path, dataset: str, config: dict) -> tuple[Path, list[dict]]:
    root = work_root / "acquisition" / dataset / "source"
    names = [config["source_cgm"], *config["metadata_sha256"]]
    records = []
    for name in names:
        path = root / name
        expected_hash = config["source_cgm_sha256"] if name == config["source_cgm"] else config["metadata_sha256"][name]
        expected_size = config["source_cgm_size_bytes"] if name == config["source_cgm"] else path.stat().st_size
        checked_file(path, expected_hash, expected_size)
        records.append({"source_file": name, "local_file": str(path), "size_bytes": path.stat().st_size,
                        "sha256": expected_hash, "source_uri": f"source://{dataset}/" +
                        ("Data Files/" if dataset == "Brown_2019" else "Data Tables/") + name})
    return root, records


def read_table(root: Path, name: str, brown: bool = False) -> pd.DataFrame:
    return pd.read_csv(root / name, sep="|", encoding="utf-16" if brown else "utf-8-sig",
                       dtype="string", keep_default_na=False)


def validate_brown_treatment_audit(screening: pd.DataFrame, config: dict) -> None:
    """Fail if an adult screened 670G user is absent from the conservative exclusion."""
    screened_670g = set(screening.loc[
        screening.PumpType.str.contains("670G", case=False, regex=False)
        & pd.to_numeric(screening.AgeAtEnrollment, errors="raise").ge(18), "PtID"])
    unverified = set(config["unverified_baseline_treatment_ids"])
    if screened_670g != unverified or not unverified.isdisjoint(config["unverifiable_baseline_clock_ids"]):
        raise ValueError("Brown screened 670G participants differ from configured treatment-mode audit")


def metadata(root: Path, dataset: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset == "Anderson_2016":
        enrollment = read_table(root, "Enrollment.txt")
        visits = read_table(root, "VisitInfo.txt")
        if len(enrollment) != 30 or enrollment.DeidentID.duplicated().any() or set(enrollment.SubEligibility) != {"1"}:
            raise ValueError("Anderson enrollment changed")
        people = pd.DataFrame({"source_participant_id": enrollment.DeidentID,
                               "age_years": pd.to_numeric(enrollment["Age at Enrollment"], errors="raise")})
        for source, target in ((config["baseline_start_visit"], "baseline_start"),
                               (config["baseline_end_visit"], "baseline_end")):
            x = visits.loc[visits.Visit.eq(source), ["DeidentID", "VisitDt"]]
            if len(x) != len(people) or x.DeidentID.duplicated().any():
                raise ValueError(f"Anderson visit coverage changed: {source}")
            people[target] = pd.to_datetime(people.source_participant_id.map(dict(zip(x.DeidentID, x.VisitDt))), errors="raise")
        if (people.baseline_start >= people.baseline_end).any():
            raise ValueError("Anderson baseline visits out of order")
        events = read_table(root, "AdvEvent.txt").rename(columns={"DeidentID": "source_participant_id"})
    elif dataset == "Brown_2019":
        roster = read_table(root, "PtRoster_a.txt", True)
        screening = read_table(root, "DiabScreening_a.txt", True)
        if len(roster) != 170 or len(screening) != 170 or roster.PtID.duplicated().any() or screening.PtID.duplicated().any() or set(roster.PtID) != set(screening.PtID):
            raise ValueError("Brown roster or screening changed")
        people = roster[["PtID", "RandDt"]].rename(columns={"PtID": "source_participant_id"}).copy()
        people["age_years"] = pd.to_numeric(people.source_participant_id.map(
            dict(zip(screening.PtID, screening.AgeAtEnrollment))), errors="raise")
        people["baseline_start"] = pd.NaT
        people["baseline_end"] = pd.to_datetime(people.RandDt.replace("", pd.NA), errors="coerce", format="mixed")
        people["baseline_treatment"] = "pre-randomization protocol run-in or personal usual care"
        validate_brown_treatment_audit(screening, config)
        events = read_table(root, "AdvEvent_a.txt", True).rename(columns={"PtID": "source_participant_id"})
    else:
        roster = read_table(root, "IOBP2PtRoster.txt")
        baseline = read_table(root, "IOBP2RandBaseInfo.txt")
        eligibility = read_table(root, "IOBP2Eligibility.txt")
        if len(roster) != 440 or roster.PtID.duplicated().any() or baseline.PtID.duplicated().any() or eligibility.PtID.duplicated().any() or set(roster.PtID) != set(eligibility.PtID) or len(baseline) != 388:
            raise ValueError("Lynch source metadata changed")
        if eligibility.EligSrcT1DIns1Yr.eq("").any():
            raise ValueError("Lynch T1D eligibility missing")
        people = roster[["PtID", "RandDt", "AgeAsofEnrollDt"]].rename(
            columns={"PtID": "source_participant_id"}).copy()
        people["age_years"] = pd.to_numeric(people.AgeAsofEnrollDt, errors="coerce")
        people["baseline_start"] = pd.NaT
        people["baseline_end"] = pd.to_datetime(people.RandDt, format="mixed", dayfirst=False, errors="raise")
        people["baseline_treatment"] = people.source_participant_id.map(dict(zip(baseline.PtID, baseline.DiabTrt)))
        people["baseline_cgm_type"] = people.source_participant_id.map(dict(zip(baseline.PtID, baseline.CGMType)))
        actual = set(baseline.DiabTrt)
        allowed = set(config["allowed_baseline_treatment"])
        automated = set(config["excluded_automated_baseline_treatment"])
        if actual != allowed | automated or allowed & automated:
            raise ValueError("Lynch baseline modality values differ from audited categories")
        events = read_table(root, "IOBP2AdvEvent.txt").rename(columns={"PtID": "source_participant_id"})
    if people.source_participant_id.duplicated().any() or people.age_years.isna().any():
        raise ValueError("T1D source participant or age missing")
    people["dataset"] = dataset
    people["participant_id"] = people.source_participant_id.map(lambda x: analysis_id(dataset, str(x)))
    people["source_disease_label"] = config["source_label"]
    people["primary_cohort"] = "T1D"
    people["device"] = config["device"]
    people["adult_eligible"] = people.age_years.ge(18)
    if dataset == "Lynch_2022":
        people["period_eligible"] = people.baseline_treatment.isin(config["allowed_baseline_treatment"])
        people["fixed_exclusion"] = people.source_participant_id.isin(config["prespecified_exclusions"])
    elif dataset == "Brown_2019":
        people["period_eligible"] = (people.baseline_end.notna()
                                     & ~people.source_participant_id.isin(config["unverifiable_baseline_clock_ids"])
                                     & ~people.source_participant_id.isin(config["unverified_baseline_treatment_ids"]))
        people["fixed_exclusion"] = False
    else:
        people["period_eligible"] = True
        people["fixed_exclusion"] = False
    return people, events


def select_time(chunk: pd.DataFrame, dataset: str) -> tuple[pd.Series, pd.Series]:
    if dataset == "Anderson_2016":
        corrected = chunk.DisplayTimeAdjusted.ne("")
        source = chunk.DisplayTimeAdjusted.where(corrected, chunk.DisplayTime)
        parsed = pd.to_datetime(source, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    elif dataset == "Brown_2019":
        corrected = chunk.DataDtTm_adj.ne("")
        source = chunk.DataDtTm_adj.where(corrected, chunk.DataDtTm)
        parsed = pd.to_datetime(source, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    else:
        corrected = pd.Series(False, index=chunk.index)
        source = chunk.DeviceDtTm
        parsed = pd.to_datetime(source, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    return parsed, corrected


def adverse_event_mask(rows: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.Series, int]:
    """Delete event calendar days through the resolved day, plus the next day."""
    affected = pd.Series(False, index=rows.index)
    applicable = 0
    event_people = set(rows.source_participant_id)
    for event in events.itertuples(index=False):
        if event.source_participant_id not in event_people:
            continue
        start = pd.to_datetime(event.AEOnsetDt, errors="coerce")
        end = pd.to_datetime(event.AEResDt, errors="coerce")
        if pd.isna(start):
            raise ValueError("source adverse event lacks onset date")
        last = (end if pd.notna(end) else start).normalize() + pd.Timedelta(days=1)
        if last < start.normalize():
            raise ValueError("source adverse event resolves before onset")
        affected |= rows.source_participant_id.eq(event.source_participant_id) & rows.timestamp_local.dt.normalize().between(start.normalize(), last)
        applicable += 1
    return affected, applicable


def ingest(root: Path, dataset: str, people: pd.DataFrame, events: pd.DataFrame,
           config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = root / config["source_cgm"]
    encoding = "utf-16" if dataset == "Brown_2019" else "utf-8-sig"
    reader = pd.read_csv(path, sep="|", encoding=encoding, dtype="string", keep_default_na=False, chunksize=250000)
    person = people.set_index("source_participant_id")
    id_column = "DeidentID" if dataset == "Anderson_2016" else "PtID"
    glucose_column = "Value" if dataset == "Lynch_2022" else "CGM"
    batches = []
    counts = {x: 0 for x in ("source_rows", "device_measurement_rows", "adult_rows", "disease_label_rows",
                              "period_rows", "fixed_exclusion_rows", "adverse_event_rows", "valid_rows")}
    source_ids = set()
    source_types = {}
    source_units = {}
    adjusted_rows = 0
    bad_time = 0
    for chunk in reader:
        if chunk.columns.tolist() != SOURCE_COLUMNS[dataset]:
            raise ValueError(f"unexpected {dataset} CGM schema")
        counts["source_rows"] += len(chunk)
        ids = chunk[id_column]
        source_ids.update(ids.unique())
        if not set(ids).issubset(person.index):
            raise ValueError("CGM participant absent from source metadata")
        if dataset == "Lynch_2022":
            for value, n in chunk.RecordType.value_counts().items():
                source_types[value] = source_types.get(value, 0) + int(n)
            for value, n in chunk.Units.value_counts().items():
                source_units[value] = source_units.get(value, 0) + int(n)
            if set(chunk.RecordType) - {"CGM", "Calibration", "BG"} or set(chunk.Units) != {"mg/dL"} or chunk.InternalDtTm.ne("").any():
                raise ValueError("Lynch record type, unit, or internal clock changed")
            chunk = chunk.loc[chunk.RecordType.eq("CGM")].copy()
        else:
            if dataset == "Brown_2019" and set(chunk.EventSubType) - {"", "HIGH", "LOW"}:
                raise ValueError("Brown CGM event subtype changed")
        counts["device_measurement_rows"] += len(chunk)
        chunk = chunk.loc[chunk[id_column].map(person.adult_eligible).fillna(False)].copy()
        counts["adult_rows"] += len(chunk)
        counts["disease_label_rows"] += len(chunk)  # T1D trial eligibility is source-level.
        if chunk.empty:
            continue
        time, corrected = select_time(chunk, dataset)
        invalid_time = time.isna()
        bad_time += int(invalid_time.sum())
        if invalid_time.any():
            if dataset != "Lynch_2022" or not chunk.loc[invalid_time, "DeviceDtTm"].str.fullmatch(
                    r"\d{1,2}/\d{1,2}/\d{4}").all():
                raise ValueError(f"unparseable {dataset} source CGM timestamp")
            chunk = chunk.loc[~invalid_time].copy()
            time = time.loc[~invalid_time]
            corrected = corrected.loc[~invalid_time]
        chunk["timestamp_local"] = time
        adjusted_rows += int(corrected.sum())
        starts = chunk[id_column].map(person.baseline_start)
        ends = chunk[id_column].map(person.baseline_end)
        allowed = chunk[id_column].map(person.period_eligible).fillna(False)
        in_period = allowed & time.lt(ends)
        if dataset == "Anderson_2016":
            in_period &= time.ge(starts)
        chunk = chunk.loc[in_period].copy()
        counts["period_rows"] += len(chunk)
        chunk = chunk.loc[~chunk[id_column].map(person.fixed_exclusion).fillna(False)].copy()
        counts["fixed_exclusion_rows"] += len(chunk)
        if chunk.empty:
            continue
        batches.append(pd.DataFrame({"dataset": dataset,
                                     "source_participant_id": chunk[id_column].astype(str),
                                     "participant_id": chunk[id_column].map(person.participant_id),
                                     "source_record_id": chunk.RecID.astype(str),
                                     "source_timestamp": (chunk.DisplayTime if dataset == "Anderson_2016" else
                                                          chunk.DataDtTm if dataset == "Brown_2019" else chunk.DeviceDtTm).astype(str),
                                     "source_timestamp_corrected": (chunk.DisplayTimeAdjusted if dataset == "Anderson_2016" else
                                                                    chunk.DataDtTm_adj if dataset == "Brown_2019" else "").astype(str) if dataset != "Lynch_2022" else "",
                                     "timestamp_local": chunk.timestamp_local,
                                     "source_glucose": chunk[glucose_column].astype(str)}))
    if not source_ids.issubset(set(people.source_participant_id)) or not batches:
        raise ValueError("source CGM participant coverage differs from roster or no baseline rows")
    rows = pd.concat(batches, ignore_index=True)
    ae_mask, applicable_events = adverse_event_mask(rows, events)
    counts["adverse_event_rows"] = len(rows) - int(ae_mask.sum())
    rows = rows.loc[~ae_mask].copy()
    counts["valid_rows"] = len(rows)
    if rows.participant_id.isna().any():
        raise ValueError("missing analysis participant ID")
    qc = {"stage_rows": counts, "source_participants": len(source_ids),
          "source_types": source_types, "source_units": source_units,
          "corrected_source_time_rows_among_adults": adjusted_rows,
          "unparseable_source_time_rows_among_adults": bad_time,
          "applicable_adverse_events": applicable_events,
          "adverse_event_removed_rows": int(ae_mask.sum()),
          "source_rows_by_participant": rows.source_participant_id.value_counts().sort_index().to_dict()}
    return rows, people, qc


def attach(frame: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    info = people.set_index("participant_id")
    for name in ("source_participant_id", "source_disease_label", "primary_cohort", "device", "age_years"):
        frame[name] = frame.participant_id.map(info[name])
    frame["disease_subgroup"] = "T1D"
    return frame


def clean_dataset(rows: pd.DataFrame, people: pd.DataFrame, rules: Rules) -> dict:
    dataset = str(rows.dataset.iloc[0])
    values, glucose_qc = standardize_glucose(rows.source_glucose, "mg/dL", rules)
    if values.isna().any():
        raise ValueError("nonnumeric T1D CGM glucose")
    aligned_input = rows[["dataset", "participant_id", "timestamp_local"]].copy()
    aligned_input["y_mgdl"] = values
    observed, align_qc = align_and_aggregate(aligned_input, rules)
    grid, segments = make_natural_segments(observed, rules)
    washed, wash_qc = apply_boundary_washout(grid, rules)
    washed["partition"] = "external_test"
    washed["external_dataset"] = dataset
    analysis, interpolation_qc = interpolate_short_gaps(washed, rules)
    origins, origin_qc = common_origins(analysis, rules)
    for frame in (observed, washed, analysis, origins):
        info = attach(frame, people)
        for column in ("source_participant_id", "source_disease_label", "primary_cohort", "device", "age_years", "disease_subgroup"):
            frame[column] = info[column]
    observed["partition"] = "external_test"
    observed["external_dataset"] = dataset
    assert_cleaning_qc(observed, analysis, origins, rules)
    return {"observed": observed, "segments": segments, "washed": washed,
            "analysis": analysis, "origins": origins, "glucose_qc": glucose_qc,
            "alignment_qc": align_qc, "washout_qc": wash_qc,
            "interpolation_qc": interpolation_qc, "origin_qc": origin_qc,
            "grid_positions": len(grid)}


def output_hashes(directory: Path) -> list[dict]:
    return [{"file": str(path.relative_to(directory)), "size_bytes": path.stat().st_size,
             "sha256": sha256_file(path)} for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "build_manifest.json"]


def run_dataset(work_root: Path, dataset: str) -> Path:
    config = config_for(dataset)
    pipeline = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())
    if pipeline["specification"]["version"] != "1.5":
        raise ValueError("unexpected pipeline specification")
    rules = Rules.from_config(pipeline["cleaning"])
    root, files = source_files(work_root, dataset, config)
    people, events = metadata(root, dataset, config)
    rows, people, ingestion_qc = ingest(root, dataset, people, events, config)
    built = clean_dataset(rows, people, rules)
    counts = built["origins"].groupby("participant_id").size()
    people["common_origins"] = people.participant_id.map(counts).fillna(0).astype(int)
    people["externally_evaluable"] = people.common_origins.gt(0)
    people["has_eligible_source_rows"] = people.participant_id.isin(rows.participant_id)
    if dataset == "Lynch_2022" and people.loc[people.fixed_exclusion, "has_eligible_source_rows"].any():
        raise ValueError("Lynch fixed exclusion reappeared")
    exclusions = people.loc[~people.adult_eligible | ~people.period_eligible | people.fixed_exclusion |
                            ~people.has_eligible_source_rows | ~people.externally_evaluable,
                            ["source_participant_id", "participant_id"]].copy()
    unverified_mode = (people.source_participant_id.isin(config.get("unverified_baseline_treatment_ids", []))
                       if dataset == "Brown_2019" else pd.Series(False, index=people.index))
    reason = np.select([~people.adult_eligible, unverified_mode, ~people.period_eligible, people.fixed_exclusion,
                        ~people.has_eligible_source_rows, ~people.externally_evaluable],
                       ["minor", "eligible baseline treatment mode unverified", "ineligible_or_unverifiable_baseline", "prespecified_data_quality",
                        "no_rows_after_source_exclusions", "no_common_origin"], default="eligible")
    exclusions["reason"] = pd.Series(reason, index=people.index).loc[exclusions.index]
    exclusions["prespecified_exclusion_flag"] = people.loc[exclusions.index, "fixed_exclusion"]
    qc_people = people[["source_participant_id", "participant_id", "age_years", "adult_eligible",
                        "period_eligible", "fixed_exclusion", "has_eligible_source_rows", "common_origins"]].copy()
    for key, frame in (("source_rows", rows), ("observed_points", built["observed"]),
                       ("segments", built["segments"]), ("post_washout_positions", built["analysis"])):
        source = frame.groupby("participant_id").size()
        qc_people[key] = qc_people.participant_id.map(source).fillna(0).astype(int)
    qc_people["post_washout_observed_positions"] = qc_people.participant_id.map(
        built["analysis"].loc[built["analysis"].is_observed].groupby("participant_id").size()).fillna(0).astype(int)
    qc_people["interpolated_positions"] = qc_people.participant_id.map(
        built["analysis"].loc[built["analysis"].is_interpolated].groupby("participant_id").size()).fillna(0).astype(int)
    qc_people["observed_duration_minutes"] = qc_people.post_washout_observed_positions * rules.grid_minutes
    qc_people["observed_proportion"] = qc_people.post_washout_observed_positions / qc_people.post_washout_positions.replace(0, np.nan)
    qc_people["interpolated_proportion"] = qc_people.interpolated_positions / qc_people.post_washout_positions.replace(0, np.nan)
    stage = ingestion_qc["stage_rows"]
    flow = pd.DataFrame([
        ("source_ingestion", stage["source_rows"], len(people), "source_rows"),
        ("device_measurements", stage["device_measurement_rows"], len(people), "source_rows"),
        ("adult_eligibility", stage["adult_rows"], int(people.adult_eligible.sum()), "source_rows"),
        ("disease_label_eligibility", stage["disease_label_rows"], int(people.adult_eligible.sum()), "source_rows"),
        ("usual_care_period", stage["period_rows"], int((people.adult_eligible & people.period_eligible).sum()), "source_rows"),
        ("prespecified_exclusions", stage["fixed_exclusion_rows"], int((people.adult_eligible & people.period_eligible & ~people.fixed_exclusion).sum()), "source_rows"),
        ("adverse_event_deletion", stage["adverse_event_rows"], int(people.has_eligible_source_rows.sum()), "source_rows"),
        ("valid_time_and_glucose", stage["valid_rows"], int(people.has_eligible_source_rows.sum()), "source_rows"),
        ("five_minute_alignment", len(built["observed"]), int(built["observed"].participant_id.nunique()), "observed_grid_positions"),
        ("natural_segments", built["grid_positions"], int(built["segments"].participant_id.nunique()), "grid_positions"),
        ("boundary_washout", len(built["washed"]), int(built["washed"].participant_id.nunique()), "grid_positions"),
        ("permitted_interpolation", len(built["analysis"]), int(built["analysis"].participant_id.nunique()), "grid_positions"),
        ("common_origins", len(built["origins"]), int(built["origins"].participant_id.nunique()), "origins"),
    ], columns=["stage", "rows_after", "participants_after", "row_unit"])
    qc = {"status": "PASS", "dataset": dataset, "source_rows": ingestion_qc["stage_rows"]["source_rows"],
          "source_participants": len(people), "adult_participants": int(people.adult_eligible.sum()),
          "period_eligible_adults": int((people.adult_eligible & people.period_eligible).sum()),
          "fixed_exclusions_among_period_eligible_adults": int((people.adult_eligible & people.period_eligible & people.fixed_exclusion).sum()),
          "participants_with_rows_after_exclusions": int(people.has_eligible_source_rows.sum()),
          "participants_with_origins": int(people.externally_evaluable.sum()),
          "prespecified_exclusion_ids": config.get("prespecified_exclusions", []),
          "prespecified_exclusion_count_across_all_source_people": int(people.fixed_exclusion.sum()),
          "origins": len(built["origins"]), "source_ingestion": ingestion_qc,
          "glucose_standardization": built["glucose_qc"], "alignment": built["alignment_qc"],
          "natural_segments": len(built["segments"]), "natural_grid_positions": built["grid_positions"],
          "washout": built["washout_qc"], "interpolation": built["interpolation_qc"],
          "origin_eligibility": built["origin_qc"], "normalization_from_external": False,
          "source_clock_policy": config["source_time_policy"]}
    started = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    out = work_root / "build" / dataset / run_id
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in (("source_ingestion", rows), ("cgm_observed", built["observed"]),
                        ("natural_segments", built["segments"]), ("cgm_washed", built["washed"]),
                        ("cgm_analysis", built["analysis"]), ("forecast_origins", built["origins"])):
        frame.to_parquet(out / f"{name}.parquet", index=False)
    for name, frame in (("source_cohort_manifest", people), ("source_exclusion_log", exclusions),
                        ("participant_qc", qc_people), ("qc_flow", flow), ("data_manifest", pd.DataFrame(files))):
        frame.to_csv(out / f"{name}.csv", index=False)
    write_json_new(out / "qc_report.json", qc)
    code_files = ["configs/pipeline.yaml", f"configs/datasets/{dataset.split('_')[0].lower()}.yaml",
                  "src/cgm_pipeline/build/common.py", "src/cgm_pipeline/build/t1d.py",
                  "scripts/build/build_t1d.py"]
    write_json_new(out / "build_manifest.json", {
        "dataset": dataset, "run_id": run_id, "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": files, "source_config": config, "rules": rules.__dict__,
        "code_commit_at_build": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "code_file_sha256": {name: sha256_file(REPO_ROOT / name) for name in code_files},
        "dependency_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
        "reproduction_command": DATA_BUILD_COMMAND,
        "output_files": output_hashes(out)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--work-root", type=Path)
    args = parser.parse_args()
    print(run_dataset(resolve_work_root(args.work_root), args.dataset))


if __name__ == "__main__":
    main()
