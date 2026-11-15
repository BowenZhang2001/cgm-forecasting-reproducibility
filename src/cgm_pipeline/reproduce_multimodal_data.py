"""Build CGMacros within-participant or leave-one-subject-out forecasting data."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cgm_pipeline import cgmacros_multimodal_data as prep
from cgm_pipeline.meal_responses import assess_response
from cgm_pipeline.reproduce_baselines import CONFIG


@dataclass
class CohortData:
    context: np.ndarray
    target: np.ndarray
    rows: list

    @property
    def people(self):
        return np.asarray([row["participant"] for row in self.rows])


def raw_sources(root):
    files = sorted(root.rglob("CGMacros-[0-9][0-9][0-9].csv"))
    participants = {path.stem[-3:]: path for path in files}
    if len(files) != 45 or len(participants) != 45:
        raise ValueError("Expected 45 original participant CSVs from CGMacros")
    return participants


def image_inventory(sources):
    assignments = {}
    for pid, csv_file in sources.items():
        for file in csv_file.parent.rglob("*"):
            relative = file.relative_to(csv_file.parent)
            if relative.parts[0].casefold() == "photos" and file.is_file():
                key = f"source://CGMacros/CGMacros-{pid}/{relative.as_posix()}"
                assignments[key] = {"bytes": file.stat().st_size, "path": file}
    return assignments


def review_hashes(reviews):
    if reviews is None:
        raise ValueError(
            "No CGMacros photo exclusion list was resolved. "
            "The CLI uses the bundled list unless --image-qc is specified."
        )
    with reviews.open() as stream:
        spec = json.load(stream)
    return set(spec["rejected_sha256"])


def available_photo_sha256(file, excluded):
    with file.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest in excluded:
        return None
    try:
        with Image.open(file) as image:
            image.load()
    except OSError:
        return None
    return digest


def prepare(sources, inventory, excluded, held_out=None):
    config = json.loads((CONFIG / "cgmacros_multimodal.json").read_text())
    partitions = ("train", "validation", "test")
    contexts = {k: [] for k in partitions}
    targets = {k: [] for k in partitions}
    all_rows = {k: [] for k in partitions}
    all_meals = {}
    curve_data = {}
    for pid, source in sources.items():
        points, meals, photos, _, phase = prep.read_participant(source, config)
        bins = prep.make_bins(points, phase)
        prep.segment_bins(bins, config)
        if held_out is None:
            prep.assign_splits(bins)
        else:
            prep.loso_splits(bins, pid, held_out)
        prep.context_fills(bins, config)
        canonical, _, _ = prep.canonical_meals(pid, meals, photos, bins, inventory)
        for meal in canonical:
            path = None
            sha = None
            if meal["image_candidate_selected"] and meal["partition"] in partitions:
                path = inventory[meal["image_object_uri"]]["path"]
                sha = available_photo_sha256(path, excluded)
            meal["image_sha256"] = sha or ""
            meal["image_path"] = path if sha else None
            all_meals[meal["meal_id"]] = meal
        location = {b["t"]: i for i, b in enumerate(bins)}
        origins = prep.origins(pid, bins, canonical, config)
        for row in origins:
            if not row["eligible"]:
                continue
            position = location[datetime.fromisoformat(row["origin_timestamp"])]
            part = row["partition"]
            contexts[part].append([b["v"] if b["complete"] else b["fill"]["value"]
                                   for b in bins[position - 287: position + 1]])
            targets[part].append([b["v"] for b in bins[position + 1: position + 25]])
            all_rows[part].append(row)
        test_onsets = [row for row in origins if row["eligible"] and row["partition"] == "test"
                       and row["meal_onset"]]
        if test_onsets:
            grid = pd.DataFrame([prep.bin_row(pid, b, bins) for b in bins])
            for name in ("timestamp_local", "phase_identified_at", "phase_invalid_from",
                         "conditional_dependency_time"):
                grid[name] = pd.to_datetime(grid[name])
            grid = grid.set_index("timestamp_local")
            accepted = {meal["meal_id"] for meal in canonical if meal["image_sha256"]}
            event_rows = [dict(meal, natural_segment=float(meal["natural_segment"])
                               if meal["natural_segment"] not in (None, "") else float("nan"))
                          for meal in canonical]
            event_by_id = {meal["meal_id"]: meal for meal in event_rows}
            for row in test_onsets:
                meal = event_by_id[row["onset_meal_id"]]
                result, window = assess_response(meal, grid, event_rows, accepted)
                if result["eligible"]:
                    curve_data[row["origin_id"]] = {
                        "baseline": result["baseline"],
                        "observed_peak": result["peak_rise"],
                        "observed_iauc": result["positive_iauc"],
                        "origin_glucose": window[6]["source_processed_value"],
                    }
    data = {part: CohortData(np.asarray(contexts[part], dtype=np.float32),
                             np.asarray(targets[part], dtype=np.float32), all_rows[part])
            for part in partitions}
    return data, all_meals, curve_data
