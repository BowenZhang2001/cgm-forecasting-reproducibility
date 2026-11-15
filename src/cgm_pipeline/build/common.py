"""Dataset-neutral, deterministic CGM cleaning and common-origin rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DATA_BUILD_COMMAND = "python3 run_study.py --paths /path/to/local_paths.json --stage data"


@dataclass(frozen=True)
class Rules:
    mmol_to_mgdl: float
    glucose_min: float
    glucose_max: float
    grid_minutes: int
    max_short_gap_positions: int
    washout_positions: int
    context_positions: int
    target_positions: int
    stride_positions: int

    @classmethod
    def from_config(cls, config: dict) -> "Rules":
        return cls(
            mmol_to_mgdl=float(config["glucose_mmol_to_mgdl"]),
            glucose_min=float(config["glucose_min_mgdl"]),
            glucose_max=float(config["glucose_max_mgdl"]),
            grid_minutes=int(config["grid_minutes"]),
            max_short_gap_positions=int(config["max_short_gap_positions"]),
            washout_positions=int(config["boundary_washout_positions"]),
            context_positions=int(config["context_positions"]),
            target_positions=int(config["target_positions"]),
            stride_positions=int(config["origin_stride_positions"]),
        )


def standardize_glucose(values: pd.Series, unit: str, rules: Rules) -> tuple[pd.Series, dict[str, int]]:
    """Parse, convert, then clip; unknown strings stay missing for the exclusion step."""
    if unit not in {"mg/dL", "mmol/L"}:
        raise ValueError(f"unsupported glucose unit: {unit}")
    raw = values.astype("string").str.strip()
    low_token = raw.isin(["Low", "<40"])
    high_token = raw.isin(["High", ">400"])
    numeric = pd.to_numeric(raw.where(~(low_token | high_token)), errors="coerce").astype("float64")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("nonfinite source glucose")
    if unit == "mmol/L":
        numeric *= rules.mmol_to_mgdl
    below = numeric < rules.glucose_min
    above = numeric > rules.glucose_max
    invalid = numeric.isna() & ~(low_token | high_token)
    numeric = numeric.clip(lower=rules.glucose_min, upper=rules.glucose_max)
    numeric.loc[low_token] = rules.glucose_min
    numeric.loc[high_token] = rules.glucose_max
    return numeric, {
        "input_rows": len(values), "converted_rows": int((~invalid & ~(low_token | high_token)).sum()) if unit == "mmol/L" else 0,
        "low_token_rows": int(low_token.sum()), "high_token_rows": int(high_token.sum()),
        "below_range_rows": int(below.sum()), "above_range_rows": int(above.sum()),
        "nonnumeric_rows": int(invalid.sum()),
    }


def align_and_aggregate(rows: pd.DataFrame, rules: Rules,
                        time_col: str = "timestamp_local") -> tuple[pd.DataFrame, dict[str, int]]:
    """Nearest grid point, exact half-grid ties toward the later local time."""
    required = {"dataset", "participant_id", time_col, "y_mgdl"}
    if not required.issubset(rows.columns):
        raise ValueError("missing alignment columns")
    if rows[time_col].isna().any() or rows["y_mgdl"].isna().any():
        raise ValueError("invalid source rows must be excluded before alignment")
    if time_col == "timestamp_local" and getattr(rows[time_col].dt, "tz", None) is not None:
        raise ValueError("timestamp_local must be a naive source-local clock")
    step_ns = rules.grid_minutes * 60 * 1_000_000_000
    aligned = rows.copy()
    aligned[time_col] = pd.to_datetime(
        ((rows[time_col].astype("int64") + step_ns // 2) // step_ns) * step_ns,
        utc=time_col == "timestamp_utc",
    )
    result = (aligned.groupby(["dataset", "participant_id", time_col], sort=True, as_index=False)
              .agg(y_mgdl=("y_mgdl", "median"), source_row_count=("y_mgdl", "size")))
    result["is_observed"] = True
    result["y_mgdl"] = result["y_mgdl"].astype("float64")
    return result, {"input_rows": len(rows), "observed_grid_rows": len(result),
                    "collapsed_source_rows": len(rows) - len(result),
                    "multi_source_grid_points": int((result["source_row_count"] > 1).sum())}


def make_natural_segments(observed: pd.DataFrame, rules: Rules,
                          time_col: str = "timestamp_local") -> tuple[pd.DataFrame, pd.DataFrame]:
    """A gap with >=25 missing grid positions starts a new segment."""
    groups = []
    descriptions = []
    keys = ["dataset", "participant_id"]
    step = pd.Timedelta(minutes=rules.grid_minutes)
    for (dataset, participant), group in observed.groupby(keys, sort=True):
        group = group.sort_values(time_col).reset_index(drop=True)
        if group[time_col].duplicated().any():
            raise ValueError("duplicate aligned participant-time point")
        gaps = group[time_col].diff().div(step).fillna(1).astype("int64")
        if (gaps < 1).any():
            raise ValueError("nonpositive aligned gap")
        segment_numbers = (gaps.sub(1).ge(rules.max_short_gap_positions + 1)).cumsum().add(1)
        for number, part in group.groupby(segment_numbers, sort=True):
            segment_id = f"{participant}#S{number:04d}"
            grid = pd.DataFrame({time_col: pd.date_range(
                part[time_col].iloc[0], part[time_col].iloc[-1], freq=step)})
            grid = grid.merge(part[[time_col, "y_mgdl", "source_row_count"]],
                              on=time_col, how="left", validate="one_to_one")
            grid["dataset"] = dataset
            grid["participant_id"] = participant
            grid["natural_segment_id"] = segment_id
            grid["is_observed"] = grid["source_row_count"].notna()
            grid["source_row_count"] = grid["source_row_count"].fillna(0).astype("int64")
            groups.append(grid)
            descriptions.append({"dataset": dataset, "participant_id": participant,
                                 "natural_segment_id": segment_id, "grid_positions": len(grid),
                                 "observed_positions": int(grid["is_observed"].sum()),
                                 "missing_positions": int((~grid["is_observed"]).sum()),
                                 "source_start_time": grid[time_col].iloc[0],
                                 "source_end_time": grid[time_col].iloc[-1]})
    if not groups:
        raise ValueError("no observed CGM rows")
    return pd.concat(groups, ignore_index=True), pd.DataFrame(descriptions)


def apply_boundary_washout(grid: pd.DataFrame, rules: Rules,
                           time_col: str = "timestamp_local") -> tuple[pd.DataFrame, dict[str, int]]:
    kept = []
    removed_positions = 0
    removed_observed = 0
    for _, group in grid.groupby(["dataset", "participant_id", "natural_segment_id"], sort=True):
        ordered = group.sort_values(time_col)
        discarded = ordered.iloc[:rules.washout_positions]
        removed_positions += len(discarded)
        removed_observed += int(discarded["is_observed"].sum())
        kept.append(ordered.iloc[rules.washout_positions:])
    return pd.concat(kept, ignore_index=True), {
        "grid_positions_before": len(grid), "grid_positions_removed": removed_positions,
        "observed_positions_removed": removed_observed,
        "grid_positions_after": len(grid) - removed_positions,
    }


def interpolate_short_gaps(grid: pd.DataFrame, rules: Rules,
                           time_col: str = "timestamp_local") -> tuple[pd.DataFrame, dict[str, int]]:
    """Only bounded internal gaps within dataset, participant, segment and partition."""
    output = []
    interpolated = 0
    for _, group in grid.groupby(["dataset", "participant_id", "natural_segment_id", "partition"], sort=True):
        part = group.sort_values(time_col).copy().reset_index(drop=True)
        observed = part["is_observed"].to_numpy(dtype=bool)
        values = part["y_mgdl"].to_numpy(dtype=float, copy=True)
        index = 0
        while index < len(part):
            if observed[index]:
                index += 1
                continue
            end = index
            while end < len(part) and not observed[end]:
                end += 1
            length = end - index
            if index > 0 and end < len(part) and length <= rules.max_short_gap_positions:
                if not observed[index - 1] or not observed[end]:
                    raise ValueError("interpolation anchors are not observed")
                left, right = values[index - 1], values[end]
                for offset in range(1, length + 1):
                    values[index + offset - 1] = left + (right - left) * offset / (length + 1)
                interpolated += length
            index = end
        part["y_mgdl"] = values
        part["is_interpolated"] = (~part["is_observed"]) & part["y_mgdl"].notna()
        output.append(part)
    result = pd.concat(output, ignore_index=True)
    if int(result["is_interpolated"].sum()) != interpolated:
        raise ValueError("interpolation count mismatch")
    return result, {"interpolated_positions": interpolated,
                    "remaining_missing_positions": int(result["y_mgdl"].isna().sum())}


def common_origins(analysis: pd.DataFrame, rules: Rules,
                   time_col: str = "timestamp_local",
                   phase_aligned_stride: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    """Context includes the observed origin; the next 24 positions are observed targets."""
    records = []
    candidates = 0
    rejections = {"stride": 0, "origin_unobserved": 0, "context_missing": 0, "target_unobserved": 0}
    group_keys = ["dataset", "participant_id", "natural_segment_id", "partition"]
    step = pd.Timedelta(minutes=rules.grid_minutes)
    for (dataset, participant, segment, partition), group in analysis.groupby(group_keys, sort=True):
        group = group.sort_values(time_col).reset_index(drop=True)
        timestamps = group[time_col]
        if timestamps.duplicated().any() or (timestamps.diff().dropna() != step).any():
            raise ValueError("irregular five-minute spacing within natural segment")
        phase = timestamps.iloc[0].minute % rules.grid_minutes if phase_aligned_stride else 0
        values = group["y_mgdl"].to_numpy(dtype=float)
        observed = group["is_observed"].to_numpy(dtype=bool)
        for index in range(rules.context_positions - 1, len(group) - rules.target_positions):
            candidates += 1
            origin = timestamps.iloc[index]
            clock_steps = (origin.hour * 60 + origin.minute - phase) // rules.grid_minutes
            if clock_steps % rules.stride_positions:
                rejections["stride"] += 1
                continue
            if not observed[index]:
                rejections["origin_unobserved"] += 1
                continue
            context = values[index - rules.context_positions + 1:index + 1]
            if not np.isfinite(context).all():
                rejections["context_missing"] += 1
                continue
            if not observed[index + 1:index + rules.target_positions + 1].all():
                rejections["target_unobserved"] += 1
                continue
            record = {
                "origin_id": f"{segment}@{origin.strftime('%Y%m%dT%H%M%S')}",
                "dataset": dataset, "participant_id": participant,
                "natural_segment_id": segment, "partition": partition,
                "external_dataset": group["external_dataset"].iloc[0],
                "context_positions": rules.context_positions,
                "target_positions": rules.target_positions,
            }
            suffix = "utc" if time_col == "timestamp_utc" else "local"
            record[f"timestamp_{suffix}"] = origin
            record[f"context_start_{suffix}"] = timestamps.iloc[index - rules.context_positions + 1]
            record[f"target_end_{suffix}"] = timestamps.iloc[index + rules.target_positions]
            records.append(record)
    suffix = "utc" if time_col == "timestamp_utc" else "local"
    columns = ["origin_id", "dataset", "participant_id", "natural_segment_id", "partition",
               "external_dataset", f"timestamp_{suffix}", f"context_start_{suffix}", f"target_end_{suffix}",
               "context_positions", "target_positions"]
    result = pd.DataFrame.from_records(records, columns=columns)
    if result["origin_id"].duplicated().any():
        raise ValueError("duplicate origin ID")
    return result, {"candidate_grid_positions": candidates, "eligible_origins": len(result),
                    "rejected_by_priority": rejections}


def assert_cleaning_qc(observed: pd.DataFrame, analysis: pd.DataFrame,
                       origins: pd.DataFrame, rules: Rules,
                       time_col: str = "timestamp_local", role: str = "external_test",
                       phase_aligned_stride: bool = False) -> None:
    for name, frame in (("observed", observed), ("analysis", analysis)):
        if not pd.api.types.is_numeric_dtype(frame["y_mgdl"]):
            raise ValueError(f"{name} y_mgdl is nonnumeric")
        valid = frame["y_mgdl"].dropna()
        if not np.isfinite(valid.to_numpy()).all() or not valid.between(rules.glucose_min, rules.glucose_max).all():
            raise ValueError(f"{name} y_mgdl outside fixed range")
        if frame.duplicated(["dataset", "participant_id", time_col]).any():
            raise ValueError(f"{name} has duplicate participant-time points")
    if role == "external_test":
        if (analysis["partition"] != "external_test").any() or (analysis["external_dataset"] != analysis["dataset"]).any():
            raise ValueError("external dataset was assigned a development partition")
    elif role == "development":
        if not analysis["partition"].isin(["train", "validation", "internal_test"]).all() or analysis["external_dataset"].notna().any():
            raise ValueError("development dataset has invalid partition or external identifier")
    else:
        raise ValueError("unknown dataset role")
    if (analysis["is_observed"] & analysis["y_mgdl"].isna()).any():
        raise ValueError("observed grid position has missing glucose")
    if (analysis["is_interpolated"] & analysis["is_observed"]).any():
        raise ValueError("interpolated grid position marked observed")
    if role == "external_test" and not origins.empty and ((origins["partition"] != "external_test").any() or
                              (origins["external_dataset"] != origins["dataset"]).any()):
        raise ValueError("origin entered a development partition")
    keys = ["dataset", "participant_id", "natural_segment_id", "partition"]
    origin_groups = dict(tuple(origins.groupby(keys, sort=False))) if not origins.empty else {}
    suffix = "utc" if time_col == "timestamp_utc" else "local"
    for key, frame in analysis.groupby(keys, sort=False):
        group = frame.sort_values(time_col)
        times = group[time_col].astype("int64").to_numpy()
        if len(times) > 1 and not np.all(np.diff(times) == rules.grid_minutes * 60 * 1_000_000_000):
            raise ValueError("irregular five-minute spacing within natural segment")
        eligible = origin_groups.pop(key, None)
        if eligible is None:
            continue
        origin_times = eligible[time_col].astype("int64").to_numpy()
        indices = np.searchsorted(times, origin_times)
        if np.any(indices >= len(times)) or np.any(times[indices] != origin_times):
            raise ValueError("origin timestamp missing from its natural segment")
        first = indices - rules.context_positions + 1
        last = indices + rules.target_positions
        if np.any(first < 0) or np.any(last >= len(times)):
            raise ValueError("origin context or target crosses a segment boundary")
        if (np.any(times[first] != eligible[f"context_start_{suffix}"].astype("int64").to_numpy()) or
                np.any(times[last] != eligible[f"target_end_{suffix}"].astype("int64").to_numpy())):
            raise ValueError("origin context or target boundary differs from manifest")
        observed_flags = group["is_observed"].to_numpy(dtype=bool)
        if not np.all(observed_flags[indices]):
            raise ValueError("forecast origin ends on an interpolated point")
        missing_prefix = np.r_[0, np.cumsum(~np.isfinite(group["y_mgdl"].to_numpy(dtype=float)))]
        if np.any(missing_prefix[indices + 1] != missing_prefix[first]):
            raise ValueError("forecast context contains missing glucose")
        unobserved_prefix = np.r_[0, np.cumsum(~observed_flags)]
        if np.any(unobserved_prefix[last + 1] != unobserved_prefix[indices + 1]):
            raise ValueError("forecast target contains nonobserved glucose")
        phase = group[time_col].iloc[0].minute % rules.grid_minutes if phase_aligned_stride else 0
        clock_steps = (eligible[time_col].dt.hour * 60 + eligible[time_col].dt.minute - phase) // rules.grid_minutes
        if (clock_steps % rules.stride_positions).any():
            raise ValueError("forecast origin violates fixed grid stride")
    if origin_groups:
        raise ValueError("origin refers to a missing natural segment or partition")
