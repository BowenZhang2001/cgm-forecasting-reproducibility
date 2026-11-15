"""Three T1D leave-one-dataset-out folds from frozen baseline builds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from cgm_pipeline.build.common import (
    DATA_BUILD_COMMAND, Rules, assert_cleaning_qc, common_origins, interpolate_short_gaps,
)
from cgm_pipeline.build.t1d import DATASETS, config_for, output_hashes
from cgm_pipeline.ingestion.broll import REPO_ROOT, resolve_work_root, sha256_file, write_json_new

PARTITIONS = ("train", "validation", "internal_test")


def assign_chronological_partitions(washed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut each participant's observed grid points at floor(60%) and floor(80%)."""
    groups = []
    records = []
    for participant, part in washed.groupby("participant_id", sort=True):
        part = part.sort_values("timestamp_local").copy()
        observed = part.loc[part.is_observed, "timestamp_local"]
        n = len(observed)
        if n < 3:
            part["partition"] = "train"
            records.append({"participant_id": participant, "observed_positions": n,
                            "train_observed": n, "validation_observed": 0, "internal_test_observed": 0,
                            "first_validation_local": pd.NaT, "first_internal_test_local": pd.NaT})
        else:
            a, b = int(np.floor(0.6 * n)), int(np.floor(0.8 * n))
            if not 0 < a < b < n:
                raise ValueError("invalid chronological observed split")
            first_val, first_test = observed.iloc[a], observed.iloc[b]
            part["partition"] = np.where(part.timestamp_local < first_val, "train",
                                         np.where(part.timestamp_local < first_test, "validation", "internal_test"))
            counts = part.loc[part.is_observed, "partition"].value_counts()
            if [int(counts.get(key, 0)) for key in PARTITIONS] != [a, b-a, n-b]:
                raise ValueError("partition observed counts differ from cutpoints")
            records.append({"participant_id": participant, "observed_positions": n,
                            "train_observed": a, "validation_observed": b-a, "internal_test_observed": n-b,
                            "first_validation_local": first_val, "first_internal_test_local": first_test})
        part["external_dataset"] = pd.NA
        groups.append(part)
    return pd.concat(groups, ignore_index=True), pd.DataFrame(records)


def development_build(washed: pd.DataFrame, rules: Rules) -> dict:
    partitioned, cuts = assign_chronological_partitions(washed)
    analysis, interpolation_qc = interpolate_short_gaps(partitioned, rules)
    origins, origin_qc = common_origins(analysis, rules)
    metadata = washed.set_index("participant_id")[["source_participant_id", "source_disease_label",
                                                   "primary_cohort", "disease_subgroup", "device", "age_years"]]
    metadata = metadata.loc[~metadata.index.duplicated()]
    for name in metadata:
        origins[name] = origins.participant_id.map(metadata[name])
    observed = analysis.loc[analysis.is_observed].copy()
    assert_cleaning_qc(observed, analysis, origins, rules, role="development")
    for participant, group in analysis.groupby("participant_id"):
        sequence = group.sort_values("timestamp_local")
        parts = sequence.partition.map({"train": 0, "validation": 1, "internal_test": 2}).to_numpy()
        if np.any(np.diff(parts) < 0):
            raise ValueError(f"partition order reverses for {participant}")
    return {"partitioned": partitioned, "cuts": cuts, "analysis": analysis,
            "origins": origins, "interpolation_qc": interpolation_qc, "origin_qc": origin_qc}


def validate_source_run(run: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = yaml.safe_load((run / "build_manifest.json").read_text())
    qc = yaml.safe_load((run / "qc_report.json").read_text())
    if manifest["dataset"] != dataset or qc["status"] != "PASS":
        raise ValueError("T1D source run identity or QC differs")
    for item in manifest["output_files"]:
        path = run / item["file"]
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"T1D source run output hash mismatch: {path}")
    washed = pd.read_parquet(run / "cgm_washed.parquet")
    external = pd.read_parquet(run / "forecast_origins.parquet")
    if set(washed.dataset) != {dataset} or (not external.empty and set(external.dataset) != {dataset}):
        raise ValueError("T1D source run dataset mismatch")
    if dataset == "Brown_2019":
        excluded = set(config_for(dataset)["unverified_baseline_treatment_ids"])
        log = pd.read_csv(run / "source_exclusion_log.csv", dtype={"source_participant_id": str})
        reasons = log.set_index("source_participant_id").reason.to_dict()
        if any(reasons.get(pid) != "eligible baseline treatment mode unverified" for pid in excluded):
            raise ValueError("Brown unverified baseline treatment exclusion missing")
        for frame in (washed, external):
            if set(frame.source_participant_id.astype(str)) & excluded:
                raise ValueError("Brown unverified baseline treatment participant reached fold input")
    return washed, external, qc


def run_folds(work_root: Path, source_runs: dict[str, Path]) -> Path:
    config = yaml.safe_load((REPO_ROOT / "configs/t1d_folds.yaml").read_text())
    pipeline = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())
    if set(source_runs) != set(DATASETS) or config["specification_version"] != "1.5" or pipeline["specification"]["version"] != "1.5":
        raise ValueError("T1D fold source or specification mismatch")
    rules = Rules.from_config(pipeline["cleaning"])
    frozen = {dataset: validate_source_run(Path(source_runs[dataset]), dataset) for dataset in DATASETS}
    started = datetime.now(timezone.utc).isoformat()
    fold_results = {}
    for name, mapping in config["folds"].items():
        heldout = mapping["external"]
        development = mapping["development"]
        if sorted([heldout, *development]) != sorted(DATASETS) or heldout in development:
            raise ValueError("T1D fold mapping is not leave-one-dataset-out")
        built = {dataset: development_build(frozen[dataset][0], rules) for dataset in development}
        all_analysis = pd.concat([built[d]["analysis"] for d in development], ignore_index=True)
        all_origins = pd.concat([built[d]["origins"] for d in development], ignore_index=True)
        heldout_origins = frozen[heldout][1]
        if set(all_analysis.dataset) != set(development) or set(all_origins.dataset) & {heldout} or set(heldout_origins.dataset) & set(development):
            raise ValueError("held-out dataset leaked into development")
        if set(all_analysis.participant_id) & set(heldout_origins.participant_id):
            raise ValueError("participant appears in held-out and development")
        excluded_brown = set(config_for("Brown_2019")["unverified_baseline_treatment_ids"])
        for frame in (all_analysis, all_origins, heldout_origins):
            if set(frame.loc[frame.dataset.eq("Brown_2019"), "source_participant_id"].astype(str)) & excluded_brown:
                raise ValueError("Brown unverified baseline treatment participant reached a fold")
        train = all_analysis.loc[all_analysis.partition.eq("train") & all_analysis.is_observed, "y_mgdl"]
        if len(train) < 2 or train.isna().any():
            raise ValueError("insufficient observed fold training values")
        normal = {"mean_mgdl": float(train.mean()), "std_mgdl": float(train.std(ddof=0)),
                  "observed_training_positions": len(train), "ddof": 0,
                  "source_partition": "train", "source_datasets": development,
                  "heldout_dataset": heldout, "interpolated_values_included": False}
        if normal["std_mgdl"] <= 0 or len(train) != sum(int(built[d]["cuts"].train_observed.sum()) for d in development):
            raise ValueError("fold normalization includes wrong values")
        origin_counts = all_origins.groupby(["dataset", "partition"]).size().to_dict()
        participants = all_origins.groupby(["dataset", "partition"]).participant_id.nunique().to_dict()
        cohort_rows = []
        for dataset in development:
            source = frozen[dataset][0]
            base = source[["dataset", "participant_id", "source_participant_id"]].drop_duplicates().copy()
            counts = built[dataset]["origins"].groupby(["participant_id", "partition"]).size().unstack(fill_value=0)
            for partition in PARTITIONS:
                base[f"{partition}_origins"] = base.participant_id.map(counts[partition] if partition in counts else {}).fillna(0).astype(int)
            base["any_origin"] = base[[f"{key}_origins" for key in PARTITIONS]].sum(axis=1).gt(0)
            base["internally_evaluable"] = base.validation_origins.gt(0) & base.internal_test_origins.gt(0)
            base["train_only"] = base.train_origins.gt(0) & ~base.internally_evaluable
            cohort_rows.append(base)
        cohort_manifest = pd.concat(cohort_rows, ignore_index=True)
        fold_results[name] = {"heldout": heldout, "development": development,
                              "built": built, "analysis": all_analysis, "origins": all_origins,
                              "heldout_origins": heldout_origins, "normalization": normal,
                              "cohort_manifest": cohort_manifest,
                              "origin_counts": {f"{d}:{p}": int(v) for (d, p), v in origin_counts.items()},
                              "origin_participants": {f"{d}:{p}": int(v) for (d, p), v in participants.items()}}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    out = work_root / "build" / "T1D_LODO" / run_id
    out.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for name, result in fold_results.items():
        folder = out / name
        folder.mkdir(exist_ok=False)
        result["analysis"].to_parquet(folder / "development_analysis.parquet", index=False)
        result["origins"].to_parquet(folder / "development_origins.parquet", index=False)
        result["heldout_origins"].to_parquet(folder / "external_origins.parquet", index=False)
        result["cohort_manifest"].to_csv(folder / "development_cohort_manifest.csv", index=False)
        for dataset, built in result["built"].items():
            built["cuts"].to_csv(folder / f"{dataset}_partition_cutpoints.csv", index=False)
        write_json_new(folder / "normalization_stats.json", result["normalization"])
        summary = {"status": "PASS", "heldout_dataset": result["heldout"],
                   "development_datasets": result["development"],
                   "development_origins": len(result["origins"]),
                   "external_origins": len(result["heldout_origins"]),
                   "development_origin_counts_by_dataset_partition": result["origin_counts"],
                   "development_participants_with_origins_by_dataset_partition": result["origin_participants"],
                   "external_participants_with_origins": int(result["heldout_origins"].participant_id.nunique()),
                   "development_train_only_participants": int(result["cohort_manifest"].train_only.sum()),
                   "development_internally_evaluable_participants": int(result["cohort_manifest"].internally_evaluable.sum()),
                   "normalization": result["normalization"]}
        write_json_new(folder / "qc_report.json", summary)
        summaries[name] = summary
    write_json_new(out / "qc_report.json", summaries)
    code_files = ["configs/pipeline.yaml", "configs/t1d_folds.yaml", "src/cgm_pipeline/build/common.py",
                  "src/cgm_pipeline/build/t1d.py", "src/cgm_pipeline/build/t1d_folds.py",
                  "scripts/build/build_t1d_folds.py"]
    write_json_new(out / "build_manifest.json", {"dataset": "T1D_LODO", "run_id": run_id,
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_runs": {dataset: str(source_runs[dataset]) for dataset in DATASETS},
        "source_run_manifest_sha256": {dataset: sha256_file(Path(source_runs[dataset]) / "build_manifest.json") for dataset in DATASETS},
        "fold_config": config, "rules": rules.__dict__, "reproduction_command": DATA_BUILD_COMMAND,
        "code_commit_at_build": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "code_file_sha256": {name: sha256_file(REPO_ROOT / name) for name in code_files},
        "dependency_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
        "output_files": output_hashes(out)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in DATASETS:
        parser.add_argument(f"--{dataset.split('_')[0].lower()}-run", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    args = parser.parse_args()
    runs = {dataset: getattr(args, dataset.split("_")[0].lower() + "_run") for dataset in DATASETS}
    print(run_folds(resolve_work_root(args.work_root), runs))


if __name__ == "__main__":
    main()
