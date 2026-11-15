"""Fixed, complete-window observed responses; no fitting or forecasting."""
from __future__ import annotations

import json
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

BASELINE_OFFSETS = tuple(range(-30, 0, 5))
OUTCOME_OFFSETS = tuple(range(0, 121, 5))
FLOW = (
    "invalid_alignment", "partition_boundary", "natural_segment_boundary", "washout_in_window",
    "missing_baseline_grid_points", "missing_outcome_grid_points", "incomplete_baseline_source",
    "incomplete_outcome_source", "baseline_dependency_invalid", "outcome_dependency_invalid",
    "subsequent_meal_within_actual_120", "subsequent_meal_alignment_extension_only",
)


def canonical_events(records: Iterable[Mapping]) -> list[dict]:
    """Reuse canonical IDs. Identical repeated rows are one event; conflicts fail."""
    events = {}
    for row in records:
        r = dict(row)
        for field in ("original_meal_timestamp", "aligned_meal_timestamp"):
            r[field] = pd.Timestamp(r[field])
        key = r["meal_id"]
        if key in events:
            old = events[key]
            if old["participant"] != r["participant"] or old["original_meal_timestamp"] != r["original_meal_timestamp"]:
                raise ValueError("Conflicting records for one canonical event")
        else:
            events[key] = r
    return sorted(events.values(), key=lambda r: (r["original_meal_timestamp"], r["meal_id"]))


def response_values(baseline: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    """Six baseline and 25 outcome points, including both integration endpoints."""
    b, y = np.asarray(baseline, dtype=np.float64), np.asarray(outcome, dtype=np.float64)
    if b.shape != (6,) or y.shape != (25,) or not np.isfinite(b).all() or not np.isfinite(y).all():
        raise ValueError("Exactly six finite baseline and 25 finite outcome values are required")
    mean = float(b.mean())
    return dict(baseline=mean, positive_iauc=float(np.trapezoid(np.maximum(y - mean, 0), dx=5)),
                peak_rise=float(y.max() - mean))


def subsequent_events(meal: Mapping, events: Iterable[Mapping], accepted_ids: set[str]) -> dict:
    m, g0 = pd.Timestamp(meal["original_meal_timestamp"]), pd.Timestamp(meal["aligned_meal_timestamp"])
    endpoint = g0 + pd.Timedelta(minutes=120)
    actual_endpoint = m + pd.Timedelta(minutes=120)
    participant_events = [r for r in canonical_events(events) if r["participant"] == meal["participant"] and r["meal_id"] != meal["meal_id"]]
    selected = [r for r in participant_events if m < r["original_meal_timestamp"] <= endpoint]
    actual = [r for r in selected if r["original_meal_timestamp"] <= actual_endpoint]
    extension = [r for r in selected if r["original_meal_timestamp"] > actual_endpoint]
    # Same historical aligned-time convention, now over the full canonical table.
    old_aligned = [r for r in participant_events if pd.notna(r["aligned_meal_timestamp"]) and g0 < r["aligned_meal_timestamp"] <= endpoint]
    details = [dict(meal_id=r["meal_id"], original_meal_timestamp=str(r["original_meal_timestamp"]),
                    aligned_meal_timestamp=str(r["aligned_meal_timestamp"]), accepted_image=r["meal_id"] in accepted_ids,
                    within_actual_120=r["original_meal_timestamp"] <= actual_endpoint,
                    alignment_extension_only=r["original_meal_timestamp"] > actual_endpoint) for r in selected]
    return dict(n_subsequent_meals_extended=len(selected), n_subsequent_meals_actual_120=len(actual),
                n_subsequent_meals_extension_only=len(extension), n_subsequent_meals_without_accepted_image=sum(r["meal_id"] not in accepted_ids for r in selected),
                any_subsequent_meal_extended=bool(selected), any_subsequent_meal_actual_120=bool(actual),
                any_subsequent_meal_extension_only=bool(extension), subsequent_event_details=json.dumps(details, separators=(",", ":")),
                full_canonical_old_aligned_count=len(old_aligned),
                accepted_image_old_aligned_count=sum(r["meal_id"] in accepted_ids for r in old_aligned))


def assess_response(meal: Mapping, grid: pd.DataFrame, events: Iterable[Mapping], accepted_ids: set[str]) -> tuple[dict, list[dict]]:
    """Audit every requested position. Context interpolation is never read."""
    m, g0 = pd.Timestamp(meal["original_meal_timestamp"]), pd.Timestamp(meal["aligned_meal_timestamp"])
    if pd.isna(m) or pd.isna(g0):
        raise ValueError("Accepted canonical/image identities require recorded and aligned timestamps")
    offsets = BASELINE_OFFSETS + OUTCOME_OFFSETS
    times = pd.DatetimeIndex([g0 + pd.Timedelta(minutes=k) for k in offsets])
    if not grid.index.is_unique or not grid.index.is_monotonic_increasing:
        raise ValueError("Grid must have unique chronological timestamps")
    w = grid.reindex(times)
    exists = times.isin(grid.index)
    numeric = lambda c: pd.to_numeric(w[c], errors="coerce").to_numpy(float)
    finite = np.isfinite(numeric("source_processed_value"))
    source = finite & (numeric("source_bin_complete") == 1) & (numeric("contributor_count") == 5) & (numeric("distinct_singleton_minutes") == 5)
    phase = pd.DatetimeIndex(pd.to_datetime(w.phase_identified_at))
    invalid = pd.DatetimeIndex(pd.to_datetime(w.phase_invalid_from))
    dependency = pd.DatetimeIndex(pd.to_datetime(w.conditional_dependency_time))
    certificate = phase.notna() & dependency.notna() & (phase <= times) & (dependency <= times) & (invalid.isna() | (invalid > times))
    dependent = certificate & (numeric("split_count_eligible") == 1)
    # Historical source-bin phase evidence is required by onset, matching accepted
    # causal(bin, origin), rather than by the earlier bin timestamp. Dependency
    # <= bin time establishes compatibility with its grid anchor. The split-count
    # flag measures earlier self-time availability and is retained as audit only.
    baseline_at_onset = (phase[:6] <= g0) & (dependency[:6] <= times[:6]) & (invalid[:6].isna() | (invalid[:6] > g0))
    partition_ok = bool((w.loc[exists, "partition"] == meal["partition"]).all())
    segment = meal["natural_segment"]
    segment_ok = pd.notna(segment) and bool((pd.to_numeric(w.loc[exists, "natural_segment"], errors="coerce") == segment).all())
    washout_ok = bool((pd.to_numeric(w.loc[exists, "washout"], errors="coerce") == 0).all())
    pos = int(grid.index.searchsorted(m, side="left"))
    ceiling_ok = pos < len(grid) and grid.index[pos] == g0 and g0 >= m
    follow = subsequent_events(meal, events, accepted_ids)
    flags = dict(invalid_alignment=not ceiling_ok, partition_boundary=not partition_ok,
                 natural_segment_boundary=not segment_ok, washout_in_window=not washout_ok,
                 missing_baseline_grid_points=not exists[:6].all(), missing_outcome_grid_points=not exists[6:].all(),
                 incomplete_baseline_source=not source[:6].all(), incomplete_outcome_source=not source[6:].all(),
                 baseline_dependency_invalid=not baseline_at_onset.all(),
                 outcome_dependency_invalid=not dependent[6:].all(),
                 subsequent_meal_within_actual_120=follow["any_subsequent_meal_actual_120"],
                 subsequent_meal_alignment_extension_only=follow["any_subsequent_meal_extension_only"])
    reasons = [key for key in FLOW if flags[key]]
    window_ok = not any(flags[key] for key in FLOW[:-2])
    values = response_values(numeric("source_processed_value")[:6], numeric("source_processed_value")[6:]) if window_ok else {}
    row = dict(canonical_meal_id=meal["meal_id"], participant=meal["participant"], partition=meal["partition"],
               image_sha256=meal["image_sha256"], original_meal_timestamp=str(m), aligned_meal_timestamp=str(g0),
               alignment_delay_minutes=(g0 - m).total_seconds() / 60,
               response_endpoint=str(g0 + pd.Timedelta(minutes=120)), actual_120_endpoint=str(m + pd.Timedelta(minutes=120)),
               natural_segment=segment, baseline_expected_points=6, outcome_expected_points=25,
               baseline_grid_points=int(exists[:6].sum()), outcome_grid_points=int(exists[6:].sum()),
               baseline_available_points=int(finite[:6].sum()), outcome_available_points=int(finite[6:].sum()),
               baseline_source_complete_points=int(source[:6].sum()), outcome_source_complete_points=int(source[6:].sum()),
               baseline_dependency_valid_points=int(baseline_at_onset.sum()), outcome_dependency_valid_points=int(dependent[6:].sum()),
               baseline_pointwise_split_count_eligible_points=int(dependent[:6].sum()),
               baseline_dependency_available_at_onset=bool(baseline_at_onset.all()),
               same_partition_present_points=partition_ok, same_segment_present_points=segment_ok,
               washout_free_present_points=washout_ok, accepted_ceiling_alignment=ceiling_ok,
               complete_valid_isolated_window=window_ok, **follow, **{"exclude_" + k: v for k, v in flags.items()},
               eligible=not reasons, exclusion_reasons="|".join(reasons), first_exclusion_reason=reasons[0] if reasons else "eligible",
               baseline=values.get("baseline", np.nan) if not reasons else np.nan,
               positive_iauc=values.get("positive_iauc", np.nan) if not reasons else np.nan,
               peak_rise=values.get("peak_rise", np.nan) if not reasons else np.nan,
               diagnostic_window_baseline=values.get("baseline", np.nan),
               diagnostic_window_positive_iauc=values.get("positive_iauc", np.nan),
               diagnostic_window_peak_rise=values.get("peak_rise", np.nan))
    points = [dict(canonical_meal_id=meal["meal_id"], participant=meal["participant"], role="baseline" if j < 6 else "outcome",
                   offset_minutes=k, timestamp_local=str(times[j]), grid_exists=bool(exists[j]),
                   source_processed_value=numeric("source_processed_value")[j], source_complete=bool(source[j]),
                   point_dependency_valid=bool(dependent[j]), partition=w.partition.iloc[j], natural_segment=w.natural_segment.iloc[j],
                   dependency_valid_at_analysis_reference=bool(baseline_at_onset[j]) if j < 6 else bool(dependent[j]),
                   washout=w.washout.iloc[j], phase_identified_at=str(phase[j]), phase_invalid_from=str(invalid[j]),
                   conditional_dependency_time=str(dependency[j]), split_count_eligible=numeric("split_count_eligible")[j])
              for j, k in enumerate(offsets)]
    return row, points
