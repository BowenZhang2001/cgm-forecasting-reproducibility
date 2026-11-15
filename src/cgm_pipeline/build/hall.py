"""Verified Hall 2018 source adapter for the common external build."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3

import pandas as pd
import yaml

from cgm_pipeline.build.common import Rules
from cgm_pipeline.build.external import save_external
from cgm_pipeline.ingestion.broll import (
    REPO_ROOT, analysis_id, checked_file, resolve_work_root,
)

DATASET = "Hall_2018"
SOURCE_COLUMNS = ["DisplayTime", "GlucoseValue", "subjectId", "InternalTime"]
DIAGNOSES = {"diabetic", "pre-diabetic", "non-diabetic"}


def map_hall_labels(clinical: pd.DataFrame, labels: dict) -> pd.DataFrame:
    if set(labels) != DIAGNOSES or set(clinical.diagnosis.dropna()) != DIAGNOSES or clinical.diagnosis.isna().any():
        raise ValueError("Hall diagnosis labels differ from explicit source values")
    result = clinical.copy()
    result["cohort_key"] = result.diagnosis.map({key: value["cohort"] for key, value in labels.items()})
    result["disease_subgroup"] = result.diagnosis.map({key: value["subgroup"] for key, value in labels.items()})
    if result[["cohort_key", "disease_subgroup"]].isna().any().any():
        raise ValueError("unmapped Hall diagnosis")
    return result


def read_source(work_root: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict]:
    if (config.get("dataset") != DATASET or config.get("source_time_column") != "DisplayTime" or
            config.get("original_glucose_unit") != "mg/dL" or set(config.get("labels", {})) != DIAGNOSES):
        raise ValueError("unexpected Hall source configuration")
    folder = work_root / "acquisition" / DATASET / DATASET
    records = []
    for kind, name in (("cgm", config["source_cgm"]), ("clinical", config["source_clinical"])):
        path = folder / name
        item = checked_file(path, config[f"source_{kind}_sha256"], config[f"source_{kind}_size_bytes"])
        records.append({"kind": kind, "source_uri": f"source://{DATASET}/{name}", **item})
    cgm = pd.read_csv(folder / config["source_cgm"], sep="\t", dtype="string", keep_default_na=False)
    if cgm.columns.tolist() != SOURCE_COLUMNS or len(cgm) != 105426:
        raise ValueError("Hall source CGM schema or row count changed")
    if cgm[SOURCE_COLUMNS].eq("").any().any():
        raise ValueError("missing Hall source CGM field")
    connection = sqlite3.connect(f"file:{folder / config['source_clinical']}?immutable=1", uri=True)
    try:
        clinical = pd.read_sql_query('SELECT userID, Age, diagnosis FROM clinical', connection)
    finally:
        connection.close()
    if (len(clinical) != 57 or clinical.userID.duplicated().any() or
            set(clinical.userID) != set(cgm.subjectId) or set(clinical.diagnosis) != DIAGNOSES):
        raise ValueError("Hall clinical IDs or exact diagnosis values changed")
    ages = pd.to_numeric(clinical.Age, errors="raise")
    if ages.isna().any() or (ages < 18).any():
        raise ValueError("Hall adult eligibility cannot be established")
    display = pd.to_datetime(cgm.DisplayTime, format="%Y-%m-%d %H:%M:%S", errors="raise")
    internal = pd.to_datetime(cgm.InternalTime, format="%Y-%m-%d %H:%M:%S", errors="raise")
    if display.isna().any() or internal.isna().any():
        raise ValueError("missing Hall source timestamp")
    clinical["source_participant_id"] = clinical.userID.astype(str)
    clinical["participant_id"] = clinical.source_participant_id.map(lambda value: analysis_id(DATASET, value))
    clinical["source_disease_label"] = clinical.diagnosis
    clinical = map_hall_labels(clinical, config["labels"])
    clinical["age_years"] = ages
    clinical["age_band"] = ages.map(lambda age: "18-39" if age < 40 else "40+")
    people = clinical[["source_participant_id", "participant_id", "source_disease_label",
                       "disease_subgroup", "cohort_key", "age_years", "age_band"]].copy()
    source = pd.DataFrame({"source_row_number": range(1, len(cgm) + 1),
                           "source_participant_id": cgm.subjectId.astype(str),
                           "participant_id": cgm.subjectId.map(dict(zip(people.source_participant_id, people.participant_id))),
                           "source_display_time": cgm.DisplayTime.astype(str),
                           "source_internal_time": cgm.InternalTime.astype(str),
                           "source_glucose": cgm.GlucoseValue.astype(str)})
    if source.participant_id.isna().any():
        raise ValueError("Hall source participant missing clinical metadata")
    numeric = pd.to_numeric(cgm.GlucoseValue, errors="coerce")
    tokens = cgm.GlucoseValue[numeric.isna()].value_counts().to_dict()
    if set(tokens) - {"Low", "High", "<40", ">400"}:
        raise ValueError("unexpected Hall glucose token")
    qc = {"source_rows": len(source), "source_participants": len(people),
          "source_columns": SOURCE_COLUMNS, "source_missing_counts": source.isna().sum().to_dict(),
          "source_label_participants": clinical.diagnosis.value_counts().to_dict(),
          "age_min": int(ages.min()), "age_max": int(ages.max()),
          "source_duplicate_participant_display_times": int(cgm.duplicated(["subjectId", "DisplayTime"]).sum()),
          "source_glucose_tokens": tokens, "source_numeric_glucose_min": float(numeric.min()),
          "source_numeric_glucose_max": float(numeric.max()),
          "display_time_min": str(display.min()), "display_time_max": str(display.max()),
          "internal_time_min": str(internal.min()), "internal_time_max": str(internal.max()),
          "timestamp_policy": config["source_time_policy"], "utc_offset": None,
          "source_rows_by_participant": source.source_participant_id.value_counts().sort_index().to_dict()}
    return source, people, records, qc


def run(work_root: Path) -> Path:
    config = yaml.safe_load((REPO_ROOT / "configs/datasets/hall.yaml").read_text())
    pipeline = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())
    if pipeline["specification"]["version"] != "1.5":
        raise ValueError("unexpected cleaning specification")
    source, people, records, source_qc = read_source(work_root, config)
    return save_external(work_root, DATASET, source, people, records, config,
                         Rules.from_config(pipeline["cleaning"]),
                         ["configs/datasets/hall.yaml", "configs/pipeline.yaml",
                          "src/cgm_pipeline/build/common.py", "src/cgm_pipeline/build/external.py",
                          "src/cgm_pipeline/build/hall.py", "scripts/build/build_hall.py"], source_qc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path)
    args = parser.parse_args()
    print(run(resolve_work_root(args.work_root)))


if __name__ == "__main__":
    main()
