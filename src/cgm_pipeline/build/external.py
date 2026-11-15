"""Common, source-independent external CGM build and audit artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from uuid import uuid4

import pandas as pd

from cgm_pipeline.build.common import (
    DATA_BUILD_COMMAND, Rules, align_and_aggregate, apply_boundary_washout, assert_cleaning_qc,
    common_origins, interpolate_short_gaps, make_natural_segments, standardize_glucose,
)
from cgm_pipeline.ingestion.broll import REPO_ROOT, sha256_file, write_json_new


def filter_causal_source_processed_origins(analysis: pd.DataFrame, origins: pd.DataFrame,
                                            rules: Rules) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    """Audit every source-processed candidate under the documented interpolation model."""
    eligible_dependencies = {}
    audit_records = []
    keys = ["dataset", "participant_id", "natural_segment_id", "partition"]
    for key, group in analysis.groupby(keys, sort=True):
        group = group.sort_values("timestamp_local").reset_index(drop=True)
        times = group.timestamp_local
        origin_group = origins.loc[
            origins.dataset.eq(key[0]) & origins.participant_id.eq(key[1]) &
            origins.natural_segment_id.eq(key[2]) & origins.partition.eq(key[3])]
        for origin in origin_group.itertuples():
            index = times.searchsorted(origin.timestamp_local)
            context = group.iloc[index - rules.context_positions + 1:index + 1]
            if len(context) != rules.context_positions:
                raise ValueError("candidate context length changed during causal audit")
            endpoint = context.iloc[-1]
            prior = context.iloc[:-1]
            endpoint_unknown = not bool(endpoint.native_dependency_inferred)
            prior_unknown = not bool(prior.native_dependency_inferred.all())
            endpoint_future = bool(endpoint.native_dependency_inferred and
                                   endpoint.latest_inferred_native_dependency_local > origin.timestamp_local)
            prior_future = bool(prior.latest_inferred_native_dependency_local.gt(origin.timestamp_local).any())
            phase_unconfirmed = bool(context.phase_confirmed_by_local_time.isna().any() or
                                     context.phase_confirmed_by_local_time.gt(origin.timestamp_local).any())
            reason = ("endpoint_future_dependency" if endpoint_future else
                      "endpoint_unknown_dependency" if endpoint_unknown else
                      "prior_context_future_dependency" if prior_future else
                      "prior_context_unknown_dependency" if prior_unknown else
                      "phase_unconfirmed_by_origin" if phase_unconfirmed else "eligible")
            audit_records.append({"origin_id": origin.origin_id, "dataset": key[0],
                "participant_id": key[1], "natural_segment_id": key[2],
                "timestamp_local": origin.timestamp_local, "decision": reason,
                "endpoint_future_dependency": endpoint_future,
                "endpoint_unknown_dependency": endpoint_unknown,
                "prior_context_future_dependency": prior_future,
                "prior_context_unknown_dependency": prior_unknown,
                "phase_unconfirmed_by_origin": phase_unconfirmed})
            if reason == "eligible":
                eligible_dependencies[origin.origin_id] = context.latest_inferred_native_dependency_local.max()
    audit = pd.DataFrame.from_records(audit_records)
    if len(audit) != len(origins):
        raise ValueError("candidate causal audit row count mismatch")
    result = origins.loc[origins.origin_id.isin(eligible_dependencies)].copy()
    result["latest_context_inferred_native_dependency_local"] = result.origin_id.map(eligible_dependencies)
    result["causal_input_model_checked"] = True
    return result, {"source_processed_candidate_origins": len(origins),
                    "removed_by_causal_input_audit": len(origins) - len(result),
                    "eligible_origins": len(result),
                    "causal_audit_decisions": audit.decision.value_counts().to_dict()}, audit


def build_cohort(source: pd.DataFrame, people: pd.DataFrame, dataset: str,
                 rules: Rules, source_processed: bool = False) -> dict:
    """Apply the fixed external rules to one explicitly labeled cohort."""
    ids = set(people.participant_id)
    rows = source.loc[source.participant_id.isin(ids)].copy()
    if rows.empty or not ids.issubset(set(rows.participant_id)):
        raise ValueError("eligible participant has no CGM rows")
    if source_processed:
        if (rows.duplicated(["participant_id", "timestamp_local"]).any() or
                rows.y_mgdl.isna().any() or not rows.timestamp_local.dt.second.eq(0).all() or
                not rows.source_row_count.eq(5).all() or
                not rows.timestamp_local.dt.minute.mod(rules.grid_minutes).eq(rows.source_phase_minutes).all()):
            raise ValueError("invalid source-processed five-minute bins")
        observed = rows[["dataset", "participant_id", "timestamp_local", "y_mgdl",
                         "source_row_count", "latest_inferred_native_dependency_local",
                         "native_dependency_inferred", "phase_confirmed_by_local_time"]].copy()
        observed["is_observed"] = True  # Available complete source-processed bin, not a native reading.
        observed["is_native_observation"] = False
        observed["source_value_provenance"] = "source_processed_five_minute_mean"
        glucose_qc = {"input_rows": len(rows), "source_processed_complete_bins": len(rows)}
        alignment_qc = {"input_rows": len(rows), "observed_grid_rows": len(rows),
                        "collapsed_source_rows": 0, "multi_source_grid_points": 0}
    else:
        values, glucose_qc = standardize_glucose(rows.source_glucose, "mg/dL", rules)
        if values.isna().any():
            raise ValueError("unexpected unparseable source glucose")
        input_rows = pd.DataFrame({"dataset": dataset, "participant_id": rows.participant_id,
                                   "timestamp_local": pd.to_datetime(rows.source_display_time, errors="raise"),
                                   "y_mgdl": values})
        observed, alignment_qc = align_and_aggregate(input_rows, rules)
    grid, segments = make_natural_segments(observed, rules)
    if source_processed:
        phase_counts = grid.groupby(["participant_id", "natural_segment_id"]).timestamp_local.agg(
            lambda value: value.dt.minute.mod(rules.grid_minutes).nunique())
        if phase_counts.gt(1).any():
            raise ValueError("multiple Dexcom phases within one natural segment")
        grid = grid.merge(observed[["participant_id", "timestamp_local", "latest_inferred_native_dependency_local",
                                    "native_dependency_inferred", "phase_confirmed_by_local_time"]],
                          on=["participant_id", "timestamp_local"], how="left", validate="one_to_one")
        grid["native_dependency_inferred"] = grid.native_dependency_inferred.astype("boolean").fillna(False).astype(bool)
        grid["is_native_observation"] = False
        grid["source_value_provenance"] = grid.is_observed.map(
            {True: "source_processed_five_minute_mean", False: "missing_grid_position"})
    partitioned, washout_qc = apply_boundary_washout(grid, rules)
    partitioned["partition"] = "external_test"
    partitioned["external_dataset"] = dataset
    analysis, interpolation_qc = interpolate_short_gaps(partitioned, rules)
    if source_processed:
        analysis["source_value_provenance"] = analysis.source_value_provenance.mask(
            analysis.is_interpolated, "pipeline_interpolation")
        for _, indices in analysis.groupby(["dataset", "participant_id", "natural_segment_id", "partition"]).groups.items():
            ordered = analysis.loc[indices].sort_values("timestamp_local")
            right = ordered.latest_inferred_native_dependency_local.bfill()
            known = (ordered.native_dependency_inferred.astype("boolean")
                     .where(ordered.is_observed).bfill().fillna(False).astype(bool))
            fill = ordered.is_interpolated
            analysis.loc[ordered.index[fill], "latest_inferred_native_dependency_local"] = right[fill]
            analysis.loc[ordered.index[fill], "native_dependency_inferred"] = known[fill]
            phase_time = ordered.phase_confirmed_by_local_time.bfill()
            analysis.loc[ordered.index[fill], "phase_confirmed_by_local_time"] = phase_time[fill]
    origins, origin_qc = common_origins(analysis, rules, phase_aligned_stride=source_processed)
    if source_processed:
        origins, causal_qc, causal_audit = filter_causal_source_processed_origins(analysis, origins, rules)
        origin_qc.update(causal_qc)
        if origins.empty:
            raise ValueError("no CGMacros origins pass the causal-input audit")
    metadata = people.set_index("participant_id")
    columns = ["source_participant_id", "source_disease_label", "disease_subgroup",
               "cohort_key", "age_years", "age_band"]
    columns += [name for name in ("device", "hba1c_percent") if name in people.columns]
    for frame in (observed, partitioned, analysis, origins):
        for column in columns:
            frame[column] = frame.participant_id.map(metadata[column])
        frame["primary_cohort"] = frame.cohort_key.map({"primary": "Prediabetes & T2D",
                                                        "healthy_supplementary": "Healthy"})
    observed["partition"] = "external_test"
    observed["external_dataset"] = dataset
    assert_cleaning_qc(observed, analysis, origins, rules, phase_aligned_stride=source_processed)
    count = origins.groupby("participant_id").size().to_dict()
    cohort = people.copy()
    cohort["common_origin_count"] = cohort.participant_id.map(count).fillna(0).astype("int64")
    cohort["externally_evaluable"] = cohort.common_origin_count.gt(0)
    cohort["partition"] = "external_test"
    cohort["external_dataset"] = dataset
    excluded = cohort.loc[~cohort.externally_evaluable, ["participant_id"]].copy()
    excluded["stage"] = "common_origins"
    excluded["reason"] = "insufficient_eligible_history_or_observed_target"
    excluded["removed_source_rows"] = 0
    participant_qc = (analysis.groupby("participant_id").agg(
        post_washout_grid_positions=("is_observed", "size"),
        post_washout_observed_positions=("is_observed", "sum"),
        interpolated_positions=("is_interpolated", "sum"),
        natural_segments=("natural_segment_id", "nunique")).reset_index())
    participant_qc["source_participant_id"] = participant_qc.participant_id.map(metadata.source_participant_id)
    participant_qc["common_origins"] = participant_qc.participant_id.map(count).fillna(0).astype("int64")
    if source_processed:
        participant_qc["candidate_origins"] = participant_qc.participant_id.map(
            causal_audit.groupby("participant_id").size()).fillna(0).astype("int64")
        for reason in ("endpoint_future_dependency", "endpoint_unknown_dependency",
                       "prior_context_future_dependency", "prior_context_unknown_dependency",
                       "phase_unconfirmed_by_origin"):
            counts = causal_audit.loc[causal_audit.decision.eq(reason)].groupby("participant_id").size()
            participant_qc[reason] = participant_qc.participant_id.map(counts).fillna(0).astype("int64")
    participant_qc["observed_duration_minutes"] = participant_qc.post_washout_observed_positions * rules.grid_minutes
    participant_qc["observed_proportion"] = participant_qc.post_washout_observed_positions / participant_qc.post_washout_grid_positions
    participant_qc["interpolated_proportion"] = participant_qc.interpolated_positions / participant_qc.post_washout_grid_positions
    flow = [
        ("source_ingestion", len(rows), len(people), "source_rows"),
        ("adult_and_label_eligibility", len(rows), len(people), "source_rows"),
        ("glucose_standardization", len(rows), len(people), "source_rows"),
        ("five_minute_alignment", len(observed), observed.participant_id.nunique(), "observed_grid_positions"),
        ("natural_grid", len(grid), grid.participant_id.nunique(), "grid_positions"),
        ("boundary_washout", len(partitioned), partitioned.participant_id.nunique(), "grid_positions"),
        ("short_gap_interpolation", len(analysis), analysis.participant_id.nunique(), "grid_positions"),
        ("common_origins", len(origins), origins.participant_id.nunique(), "origins"),
    ]
    summary = pd.DataFrame(flow, columns=["stage", "rows_after", "participants_after", "row_unit"])
    summary["rows_before"] = summary.rows_after.shift().fillna(len(rows)).astype(int)
    summary["participants_before"] = summary.participants_after.shift().fillna(len(people)).astype(int)
    qc = {"status": "PASS", "dataset": dataset, "role": "whole_dataset_external_test",
          "cohort_key": people.cohort_key.iloc[0], "source_rows": len(rows),
          "source_participants": len(people), "label_participants": people.source_disease_label.value_counts().to_dict(),
          "subgroup_participants": people.disease_subgroup.value_counts().to_dict(),
          "age_bands": people.age_band.value_counts().to_dict(),
          "glucose_standardization": glucose_qc, "alignment": alignment_qc,
          "natural_segments": len(segments), "natural_grid_positions": len(grid),
          "natural_grid_missing_positions": int((~grid.is_observed).sum()),
          "boundary_washout": washout_qc, "interpolation": interpolation_qc,
          "origin_eligibility": origin_qc, "externally_evaluable_participants": int(cohort.externally_evaluable.sum()),
          "origins_by_source_participant": dict(zip(cohort.source_participant_id,
                                                     cohort.common_origin_count.astype(int))),
          "normalization_estimated_from_external": False,
          "source_processed_target": source_processed,
          "partitions": ["external_test"], "applicable_scientific_stop_conditions": "all passed"}
    return {"observed": observed, "partitioned": partitioned, "analysis": analysis,
            "origins": origins, "cohort": cohort, "exclusions": excluded,
            "participant_qc": participant_qc, "qc_summary": summary, "qc": qc,
            "causal_audit": causal_audit if source_processed else None}


def save_external(work_root: Path, dataset: str, source: pd.DataFrame,
                  people: pd.DataFrame, source_files: list[dict], config: dict,
                  rules: Rules, code_paths: list[str], source_qc: dict,
                  source_exclusions: pd.DataFrame | None = None,
                  source_processed: bool = False,
                  bin_provenance: pd.DataFrame | None = None) -> Path:
    """Create a new, never-overwritten versioned run after source validation."""
    started = datetime.now(timezone.utc).isoformat()
    results = {key: build_cohort(source, group.copy(), dataset, rules, source_processed)
               for key, group in people.groupby("cohort_key", sort=True)}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = work_root / "build" / dataset / run_id
    directory.mkdir(parents=True, exist_ok=False)
    source.to_parquet(directory / ("five_minute_bins.parquet" if source_processed else "source_ingestion.parquet"), index=False)
    if bin_provenance is not None:
        bin_provenance.to_parquet(directory / "five_minute_bin_provenance.parquet", index=False)
    people.to_csv(directory / "source_cohort_manifest.csv", index=False)
    pd.DataFrame(source_files).to_csv(directory / "data_manifest.csv", index=False)
    write_json_new(directory / "source_ingestion_qc.json", source_qc)
    if source_exclusions is not None:
        source_exclusions.to_csv(directory / "source_exclusion_log.csv", index=False)
    for key, result in results.items():
        folder = directory / key
        folder.mkdir(exist_ok=False)
        if result["causal_audit"] is not None:
            result["causal_audit"].to_parquet(folder / "origin_causal_audit.parquet", index=False)
        for name in ("observed", "partitioned", "analysis", "origins"):
            filename = {"observed": "cgm_observed.parquet", "partitioned": "cgm_partitioned.parquet",
                        "analysis": "cgm_analysis.parquet", "origins": "forecast_origins.parquet"}[name]
            result[name].to_parquet(folder / filename, index=False)
        for name in ("cohort", "exclusions", "participant_qc", "qc_summary"):
            filename = {"cohort": "cohort_manifest.csv", "exclusions": "exclusion_log.csv",
                        "participant_qc": "participant_qc.csv", "qc_summary": "qc_summary.csv"}[name]
            result[name].to_csv(folder / filename, index=False)
        write_json_new(folder / "qc_report.json", result["qc"])
        write_json_new(folder / "normalization_stats.json", {"status": "not_applicable_external_only",
                        "training_mean_mgdl": None, "training_std_mgdl": None})
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    manifest = {"dataset": dataset, "run_id": run_id, "started_at_utc": started,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_files": source_files, "code_commit_at_build": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
                "code_file_sha256": {name: sha256_file(REPO_ROOT / name) for name in code_paths},
                "dependency_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
                "rules": rules.__dict__, "source_config": config,
                "reproduction_command": DATA_BUILD_COMMAND,
                "output_files": [{"file": str(path.relative_to(directory)), "size_bytes": path.stat().st_size,
                                  "sha256": sha256_file(path)} for path in files]}
    write_json_new(directory / "build_manifest.json", manifest)
    return directory
