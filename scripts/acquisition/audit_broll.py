#!/usr/bin/env python3
"""Inventory the pinned Broll iGlu source without installing or cleaning it."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cgm_pipeline.acquisition import broll  # noqa: E402

DEFAULT_WORK_ROOT = REPO_ROOT.parent / "work"


def resolve_work_root(explicit=None, environment=None):
    selected = explicit or environment or DEFAULT_WORK_ROOT
    root = Path(selected).expanduser().resolve()
    if len(root.parts) < 3 or root == Path.home().resolve() or root == REPO_ROOT or root in REPO_ROOT.parents:
        raise ValueError(f"unsafe work root: {root}")
    return root


def confined_path(path, work_root, label):
    resolved = path.expanduser().resolve()
    if resolved == work_root or not resolved.is_relative_to(work_root):
        raise ValueError(f"{label} must be beneath the selected work root")
    return resolved


def read_single_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"expected one source inventory row in {path}")
    return rows[0]


def validate_inventory(row):
    expected = {
        "object_name": broll.OBJECT_NAME,
        "row_count": "13866",
        "column_count": "3",
        "column_names": "id;time;gl",
        "participant_count": "5",
        "time_tzone_attribute": "EST",
        "missing_id": "0",
        "missing_time": "0",
        "missing_gl": "0",
        "duplicated_id_time": "0",
        "glucose_min": "50",
        "glucose_max": "400",
        "source_clock_date_2015_03_08_readings": "287",
    }
    mismatches = {key: (row.get(key), value) for key, value in expected.items()
                  if row.get(key) != value}
    if mismatches:
        raise ValueError(f"Broll source inventory mismatch: {mismatches}")
    median = float(row["median_gap_minutes"])
    if not 4.9 <= median <= 5.1:
        raise ValueError(f"Broll median sampling gap mismatch: {median}")


def validate_participant_inventory(path, source_row):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(source_row["participant_count"]):
        raise ValueError("participant inventory count does not match source inventory")
    if sum(int(row["row_count"]) for row in rows) != int(source_row["row_count"]):
        raise ValueError("participant row counts do not match source inventory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, help="output root; otherwise CGM_FORECAST_WORK_ROOT or repository sibling work")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-tarball", type=Path)
    source.add_argument("--download-dir", type=Path)
    parser.add_argument("--audit-root", type=Path)
    args = parser.parse_args()
    try:
        work_root = resolve_work_root(args.work_root, os.environ.get("CGM_FORECAST_WORK_ROOT"))
        audit_root = confined_path(args.audit_root or work_root / "acquisition/Broll_2021/audit", work_root, "--audit-root")
        download_dir = confined_path(args.download_dir, work_root, "--download-dir") if args.download_dir else None
    except ValueError as error:
        parser.error(str(error))
    download = broll.publish_verified_download(download_dir) if download_dir else None
    tarball = download.path if download else args.source_tarball or work_root / "acquisition/Broll_2021/source" / broll.FILENAME
    members, contents = broll.inspect_archive(tarball)
    evidence = broll.package_evidence(contents)
    now = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = audit_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    rda_path = run_dir / "example_data_5_subject.rda"
    with rda_path.open("xb") as output:
        output.write(contents[broll.RDA_MEMBER])
    (run_dir / "tar_members.txt").write_text("\n".join(members) + "\n", encoding="utf-8")
    r_script = REPO_ROOT / "scripts" / "inventory" / "inventory_broll.R"
    subprocess.run(["Rscript", "--vanilla", str(r_script), str(rda_path), str(run_dir)], check=True)
    inventory = read_single_csv(run_dir / "source_inventory.csv")
    validate_inventory(inventory)
    validate_participant_inventory(run_dir / "participant_inventory.csv", inventory)
    timezone_evidence = next(item for item in evidence if item["field"] == "timezone")
    timezone_evidence["value"] = f"{inventory['time_tzone_attribute']} (stored R time tzone attribute)"
    timezone_evidence["status"] = "source attribute verified"
    broll.write_csv(run_dir / "package_evidence.csv",
                    ("field", "value", "status", "evidence_member", "note"), evidence)
    evidence_by_field = {item["field"]: item for item in evidence}
    shared = dict(dataset="Broll_2021", package=broll.PACKAGE, version=broll.VERSION,
                  logical_source="CRAN iGlu example_data_5_subject", validated_at_utc=now,
                  source_timezone_evidence=evidence_by_field["timezone"]["value"],
                  unit_evidence=evidence_by_field["glucose_unit"]["value"],
                  disease_label_evidence=evidence_by_field["disease_group"]["value"],
                  device_evidence=evidence_by_field["device"]["value"],
                  software_license=evidence_by_field["software_license"]["value"],
                  data_license_status=evidence_by_field["data_reuse_license"]["value"],
                  adult_eligibility_status=evidence_by_field["adult_eligibility"]["value"],
                  eligible_age_range_years=evidence_by_field["eligible_age_range_years"]["value"],
                  participant_level_ages=evidence_by_field["participant_level_ages"]["value"],
                  adult_evidence_type=evidence_by_field["adult_eligibility"]["status"],
                  disease_label_scope=evidence_by_field["disease_label_scope"]["value"],
                  oral_medication_status=evidence_by_field["oral_medication_status"]["value"],
                  parent_study_insulin_therapy=evidence_by_field["parent_study_insulin_therapy"]["value"],
                  parent_study_osa_selection=evidence_by_field["parent_study_osa_selection"]["value"],
                  five_subject_selection_mechanism=evidence_by_field["five_subject_selection_mechanism"]["value"])
    manifest = [
        dict(shared, artifact_kind="package_tarball", source_url=";".join((broll.PRIMARY_URL, broll.ARCHIVE_URL)),
             member_name="", file_name=str(tarball), requested_url=download.requested_url if download else "",
             resolved_url=download.resolved_url if download else "", size_bytes=tarball.stat().st_size,
             sha256=broll.sha256_file(tarball)),
        dict(shared, artifact_kind="embedded_rda", source_url=";".join((broll.PRIMARY_URL, broll.ARCHIVE_URL)),
             member_name=broll.RDA_MEMBER, file_name=str(rda_path), requested_url=download.requested_url if download else "",
             resolved_url=download.resolved_url if download else "", size_bytes=rda_path.stat().st_size,
             sha256=broll.sha256_file(rda_path)),
    ]
    broll.write_csv(run_dir / "data_manifest.csv", broll.MANIFEST_COLUMNS, manifest)
    metadata = dict(run_id=run_id, validated_at_utc=now, source_mode="download" if download else "offline_tarball",
                    work_root=str(work_root), requested_url=download.requested_url if download else "",
                    resolved_url=download.resolved_url if download else "",
                    tarball_path=str(tarball), rda_member=broll.RDA_MEMBER, rda_path=str(rda_path),
                    python_executable=sys.executable, rscript_command="Rscript --vanilla",
                    source_timestamp_policy="preserve source-encoded fixed EST; no DST adjustment",
                    cleaning_status="not started")
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = f"""# Broll source validation

- Result: PASS; source inventory matches all prespecified source-level checks.
- Package tarball SHA256: `{manifest[0]['sha256']}`
- Embedded RDA SHA256: `{manifest[1]['sha256']}`
- Source rows: {inventory['row_count']}; participants: {inventory['participant_count']}; columns: {inventory['column_names']}.
- Missing values: id={inventory['missing_id']}, time={inventory['missing_time']}, gl={inventory['missing_gl']}.
- Duplicated id-time pairs: {inventory['duplicated_id_time']}.
- Source glucose range: {inventory['glucose_min']}–{inventory['glucose_max']} mg/dL.
- Time tzone attribute: {inventory['time_tzone_attribute']}; median within-participant sampling gap: {inventory['median_gap_minutes']} minutes.
- Source-clock 2015-03-08 readings: {inventory['source_clock_date_2015_03_08_readings']}.
- Package software license: {shared['software_license']}; separate data license: {shared['data_license_status']}.
- Adult eligibility: confirmed at parent-study level through linked parent-study eligibility; eligible age range 21–75 years; participant-level ages unavailable.
- Type 2 diabetes label scope: dataset/study level; oral-medication status unknown; insulin therapy excluded by parent study.
- Parent study: untreated moderate-to-severe OSA selection; five-person release-selection mechanism unknown.
- Timestamp policy: preserve fixed EST; no America/New_York reinterpretation or DST adjustment.
- Cleaning status: not started. No source values were changed.
"""
    (run_dir / "validation_report.md").write_text(report, encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
