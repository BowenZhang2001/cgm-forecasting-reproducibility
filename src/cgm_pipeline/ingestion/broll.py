"""Verified Broll Step 0–1 ingestion from the pinned R object."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORK_ROOT = REPO_ROOT.parent / "work"
DATASET = "Broll_2021"
TABLE_COLUMNS = (
    "source_row_number", "source_participant_id", "analysis_id",
    "source_time_unix_seconds", "source_time_fixed_EST", "source_gl",
)
FIXED_EST = timezone(timedelta(hours=-5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_id(dataset: str, participant: str) -> str:
    """Length-prefix both raw strings so their pairing is unambiguous."""
    if not dataset or not participant:
        raise ValueError("dataset and participant must be nonempty")
    return f"{len(dataset.encode('utf-8'))}:{dataset}{len(participant.encode('utf-8'))}:{participant}"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError(f"duplicate columns in {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"extra fields in {path}")
    return columns, rows


def one_row(path: Path) -> dict[str, str]:
    _, rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {path}")
    return rows[0]


def checked_file(path: Path, expected_hash: str, expected_size: int) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"source file missing: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != expected_size or digest != expected_hash:
        raise ValueError(f"source file size or SHA256 mismatch: {path}")
    return {"path": str(path), "size_bytes": size, "sha256": digest}


def resolve_work_root(explicit: Path | None) -> Path:
    root = (explicit or os.environ.get("CGM_FORECAST_WORK_ROOT") or DEFAULT_WORK_ROOT)
    root = Path(root).expanduser().resolve()
    if (len(root.parts) < 3 or root == Path.home().resolve() or
            root == REPO_ROOT or root in REPO_ROOT.parents or root.is_relative_to(REPO_ROOT)):
        raise ValueError(f"unsafe work root: {root}")
    return root


def load_config(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("dataset") != DATASET:
        raise ValueError("unexpected Broll configuration")
    return config


def verify_acquisition(work_root: Path, config: dict[str, object], audit_run_id: str) -> dict[str, object]:
    audit_id = audit_run_id
    if "/" in audit_id or ".." in audit_id:
        raise ValueError("unsafe acquisition audit ID")
    audit_dir = work_root / "acquisition" / DATASET / "audit" / audit_id
    tarball = work_root / "acquisition" / DATASET / "source" / str(config["tarball_name"])
    rda = audit_dir / Path(str(config["rda_member"])).name
    _, manifest_rows = read_csv(audit_dir / "data_manifest.csv")
    if len(manifest_rows) != 2 or {r["artifact_kind"] for r in manifest_rows} != {"package_tarball", "embedded_rda"}:
        raise ValueError("unexpected acquisition manifest artifacts")
    rows = {r["artifact_kind"]: r for r in manifest_rows}
    artifacts = {}
    for kind, path, expected_hash in (
        ("package_tarball", tarball, str(config["tarball_sha256"])),
        ("embedded_rda", rda, str(config["rda_sha256"])),
    ):
        row = rows[kind]
        if (row["dataset"] != DATASET or row["package"] != config["package"] or
                row["version"] != str(config["package_version"]) or
                row["sha256"] != expected_hash or Path(row["file_name"]).name != path.name):
            raise ValueError(f"acquisition manifest mismatch for {kind}")
        artifacts[kind] = checked_file(path, expected_hash, int(row["size_bytes"]))
    if rows["embedded_rda"]["member_name"] != config["rda_member"]:
        raise ValueError("acquisition RDA member mismatch")
    metadata = json.loads((audit_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("run_id") != audit_id or metadata.get("source_mode") != "offline_tarball":
        raise ValueError("acquisition audit ID mismatch")
    for filename in ("source_inventory.csv", "participant_inventory.csv"):
        if not (audit_dir / filename).is_file():
            raise ValueError(f"acquisition inventory missing: {filename}")
    return {"audit_id": audit_id, "audit_dir": audit_dir, "rda": rda,
            "artifacts": artifacts, "manifest_rows": rows, "audit_metadata": metadata}


def verify_inventory(run_dir: Path, audit_dir: Path) -> dict[str, object]:
    source = one_row(run_dir / "source_inventory.csv")
    if source != one_row(audit_dir / "source_inventory.csv"):
        raise ValueError("source inventory differs from verified acquisition audit")
    _, participants = read_csv(run_dir / "participant_inventory.csv")
    _, expected_participants = read_csv(audit_dir / "participant_inventory.csv")
    if participants != expected_participants:
        raise ValueError("participant inventory differs from verified acquisition audit")
    if (source["object_name"] != "example_data_5_subject" or source["column_names"] != "id;time;gl" or
            source["id_class"] != "factor" or source["time_class"] != "POSIXct;POSIXt" or
            source["gl_class"] != "integer" or source["time_tzone_attribute"] != "EST"):
        raise ValueError("unexpected source schema")
    return {"source": source, "participants": participants}


def verify_table(path: Path, inventory: dict[str, object]) -> dict[str, object]:
    columns, rows = read_csv(path)
    if tuple(columns) != TABLE_COLUMNS:
        raise ValueError("unexpected ingestion table schema")
    source = inventory["source"]
    if len(rows) != int(source["row_count"]):
        raise ValueError("ingestion row count differs from source inventory")
    counts: dict[str, int] = {}
    seen_pairs: set[tuple[str, int]] = set()
    glucose = []
    clocks = []
    march_8 = 0
    for number, row in enumerate(rows, start=1):
        if row["source_row_number"] != str(number):
            raise ValueError("source row order changed")
        participant = row["source_participant_id"]
        if row["analysis_id"] != analysis_id(DATASET, participant):
            raise ValueError("incorrect analysis ID")
        seconds = int(row["source_time_unix_seconds"])
        clock = datetime.fromtimestamp(seconds, FIXED_EST).strftime("%Y-%m-%d %H:%M:%S")
        if row["source_time_fixed_EST"] != clock:
            raise ValueError("source fixed-EST clock differs from epoch seconds")
        pair = (participant, seconds)
        if pair in seen_pairs:
            raise ValueError("duplicate source participant-time pair")
        seen_pairs.add(pair)
        counts[participant] = counts.get(participant, 0) + 1
        glucose.append(int(row["source_gl"]))
        clocks.append(clock)
        march_8 += clock.startswith("2015-03-08 ")
    expected_counts = {r["participant_id"]: int(r["row_count"]) for r in inventory["participants"]}
    if (counts != expected_counts or len(counts) != int(source["participant_count"]) or
            min(glucose) != int(source["glucose_min"]) or max(glucose) != int(source["glucose_max"]) or
            min(clocks) != source["source_time_min"] or max(clocks) != source["source_time_max"] or
            march_8 != int(source["source_clock_date_2015_03_08_readings"])):
        raise ValueError("ingestion table differs from verified Broll inventory")
    missing = {field: int(source[f"missing_{field}"]) for field in ("id", "time", "gl")}
    if any(missing.values()) or int(source["duplicated_id_time"]):
        raise ValueError("unexpected source missing value or duplicate")
    return {
        "status": "PASS", "stage": "source_ingestion_only",
        "source_columns": ["id", "time", "gl"],
        "source_column_types": {"id": source["id_class"], "time": source["time_class"], "gl": source["gl_class"]},
        "rows_before": int(source["row_count"]), "rows_after": len(rows),
        "participants_before": int(source["participant_count"]), "participants_after": len(counts),
        "missing_source_values": missing, "duplicate_participant_time_count": 0,
        "source_glucose_range": {"min": min(glucose), "max": max(glucose), "unit": "mg/dL"},
        "source_glucose_unit_distribution": {"mg/dL": len(rows)},
        "source_disease_label_field": None,
        "participant_disease_label_missing_count": None,
        "source_clock_range_fixed_EST": {"min": min(clocks), "max": max(clocks)},
        "source_clock_date_2015_03_08_rows": march_8,
        "rows_by_source_participant": counts,
        "comparison": "exact match to verified acquisition source and participant inventories",
        "transformations": ["source ID to stable analysis ID mapping only"],
    }


def write_json_new(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True, ensure_ascii=False)
        output.write("\n")


def output_record(path: Path) -> dict[str, object]:
    return {"file": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_manifest(run_dir: Path, acquisition: dict[str, object], config: dict[str, object],
                   started_at: str, finished_at: str) -> dict[str, object]:
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    code_paths = (
        "configs/datasets/broll.yaml", "src/cgm_pipeline/ingestion/broll.py",
        "scripts/ingestion/ingest_broll.py", "scripts/ingestion/export_broll.R",
        "scripts/inventory/inventory_broll.R",
    )
    audit_dir = acquisition["audit_dir"]
    return {
        "dataset": DATASET, "stage": "steps_0_1_source_manifest_and_ingestion",
        "run_id": run_dir.name, "started_at_utc": started_at, "finished_at_utc": finished_at,
        "package": config["package"], "package_version": str(config["package_version"]),
        "acquisition_audit_id": acquisition["audit_id"],
        "source_acquisition": {
            "mode": acquisition["audit_metadata"]["source_mode"],
            "download_date": None,
            "validated_at_utc": acquisition["manifest_rows"]["embedded_rda"]["validated_at_utc"],
            "documented_source_urls": [config["primary_url"], config["archive_fallback_url"]],
            "software_license": config["software_license"],
            "data_reuse_license_status": config["data_reuse_license_status"],
        },
        "code_commit": code_commit,
        "code_file_sha256": {name: sha256_file(REPO_ROOT / name) for name in code_paths},
        "dependency_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
        "source_files": acquisition["artifacts"],
        "audit_inventory_files": {name: output_record(audit_dir / name)
                                  for name in ("source_inventory.csv", "participant_inventory.csv")},
        "output_files": [output_record(run_dir / name) for name in
                         ("source_ingestion.csv", "source_inventory.csv", "participant_inventory.csv", "qc_report.json")],
        "source_metadata": {
            "original_glucose_unit": config["original_glucose_unit"],
            "device": config["device"], "timestamp_tzone_attribute": "EST",
            "timestamp_policy": config["source_timestamp_policy"],
            "disease_group": config["source_disease_label"],
            "disease_label_scope": config["source_disease_label_scope"],
            "source_disease_label_field": None,
            "adult_eligibility": config["adult_eligibility_status"],
            "parent_study_age_eligibility_years": [config["parent_study_eligible_age_min_years"],
                                                   config["parent_study_eligible_age_max_years"]],
            "individual_ages": None,
        },
        "cleaning_status": "not_started",
    }


def run(work_root: Path, *, audit_run_id: str) -> Path:
    config = load_config(REPO_ROOT / "configs/datasets/broll.yaml")
    acquisition = verify_acquisition(work_root, config, audit_run_id)
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = work_root / "ingestion" / DATASET / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    subprocess.run(["Rscript", "--vanilla", str(REPO_ROOT / "scripts/inventory/inventory_broll.R"),
                    str(acquisition["rda"]), str(run_dir)], check=True)
    subprocess.run(["Rscript", "--vanilla", str(REPO_ROOT / "scripts/ingestion/export_broll.R"),
                    str(acquisition["rda"]), str(run_dir / "source_ingestion.csv"), DATASET], check=True)
    inventory = verify_inventory(run_dir, acquisition["audit_dir"])
    qc = verify_table(run_dir / "source_ingestion.csv", inventory)
    write_json_new(run_dir / "qc_report.json", qc)
    finished_at = datetime.now(timezone.utc).isoformat()
    write_json_new(run_dir / "run_manifest.json", build_manifest(run_dir, acquisition, config, started_at, finished_at))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, help="defaults to CGM_FORECAST_WORK_ROOT or sibling work")
    parser.add_argument("--audit-run-id", required=True)
    args = parser.parse_args()
    print(run(resolve_work_root(args.work_root), audit_run_id=args.audit_run_id))


if __name__ == "__main__":
    main()
