"""CGMacros five-minute preparation, causal windows and meal eligibility."""

import bisect
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re

MINUTE = timedelta(minutes=1)
FIVE = timedelta(minutes=5)
NUTRIENTS = ("Calories", "Carbs", "Protein", "Fat", "Fiber")
CONVENTION = "source_processed_five_minute_mean"
EXCLUDED = {"024", "025", "037", "040"}
POST = re.compile(r"(?:^|[^a-z])(post(?:[-_ ]meal)?|complete[d]?|completion|leftovers?|empty[-_ ]plate)(?:[^a-z]|$)", re.I)
ROLE = re.compile(r"(?:^|[^a-z])(start|started)(?:[^a-z]|$)", re.I)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def ts(t):
    return t.isoformat(sep=" ") if t is not None else ""


def json_cell(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def table(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(raw):
    try:
        value = float(raw.strip())
        return value if math.isfinite(value) else None
    except (ValueError, AttributeError):
        return None


def glucose(raw, limits=(40, 400)):
    text = raw.strip().lower().replace(" ", "")
    if text in {"low", "<40"}:
        return float(limits[0]), "low_censored"
    if text in {"high", ">400"}:
        return float(limits[1]), "high_censored"
    value = number(raw)
    if value is None:
        return None, "blank" if not text else "invalid_or_nonfinite"
    cleaned = min(limits[1], max(limits[0], value))
    return cleaned, "range_clipped" if cleaned != value else "numeric"


def columns(headers):
    """Position-qualified collisions are retained rather than overwritten."""
    counts = Counter()
    result = []
    for position, raw in enumerate(headers):
        canonical = re.sub(r"[^a-z0-9]+", "_", raw.strip().casefold()).strip("_") or "unnamed"
        counts[canonical] += 1
        suffix = f"__{counts[canonical]}" if counts[canonical] > 1 else ""
        result.append({"position": position, "raw": raw, "canonical": canonical + suffix})
    return result


def ceiling(t, phase):
    # Minute-of-day modulo five; no .timestamp(), timezone conversion, or DST.
    base = t.replace(second=0, microsecond=0)
    if base < t:
        base += MINUTE
    return base + MINUTE * ((phase - base.minute) % 5)


def infer_run(run, tolerance):
    """First slope change identifies one phase conditionally; suffix conflicts
    invalidate only origins at/after the time the conflict becomes observable.
    Constant prefixes are unresolved. No integer-value/native-flag heuristic.
    """
    phase = known_at = invalid_at = None
    evidence = []
    for a, b, c in zip(run, run[1:], run[2:]):
        # Numeric publisher values, BEFORE clipping, establish interpolation fit.
        vals = [number(x["raw"]) for x in (a, b, c)]
        if any(v is None for v in vals):
            continue
        curvature = vals[0] - 2 * vals[1] + vals[2]
        if abs(curvature) <= tolerance:
            continue
        candidate = b["t"].minute % 5
        if phase is None:
            phase, known_at = candidate, c["t"]
            evidence = [{"source_line": x["line"], "timestamp": ts(x["t"]), "published_value": x["raw"]} for x in (a, b, c)]
        elif candidate != phase and invalid_at is None:
            invalid_at = c["t"]
    return {"phase": phase, "known_at": known_at, "invalid_at": invalid_at, "evidence": evidence,
            "start": run[0]["t"], "end": run[-1]["t"], "count": len(run)}


def read_participant(path, config):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        mapping = columns(next(reader))
        idx = {x["canonical"]: x["position"] for x in mapping}
        points, meals, photos = [], [], []
        for line, row in enumerate(reader, 2):
            if len(row) != len(mapping):
                raise ValueError(f"Malformed CSV row: {path}:{line}")
            def field(name):
                return row[idx[name]] if name in idx else ""
            t = datetime.fromisoformat(field("timestamp"))
            if t.tzinfo is not None or t.second or t.microsecond:
                raise ValueError(f"Unsupported timestamp, preserve and investigate: {path}:{line}")
            raw = field("dexcom_gl")
            value, rule = glucose(raw, config["glucose_range"])
            points.append({"t": t, "raw": raw, "v": value, "rule": rule, "line": line})
            meal_type, image = field("meal_type").strip(), field("image_path").strip()
            if meal_type:
                meals.append({"t": t, "line": line, "meal_type": meal_type, "image_reference": image,
                              "nutrients": {n: field(n.casefold()) for n in NUTRIENTS}})
            if image:
                photos.append({"t": t, "line": line, "meal_type": meal_type, "image_reference": image})
            # Other source columns are mapped, never consulted for eligibility.
    points.sort(key=lambda x: (x["t"], x["line"]))
    counts = Counter(p["t"] for p in points)
    runs, run = [], []
    for p in points:
        valid = p["v"] is not None and counts[p["t"]] == 1
        if not valid or (run and p["t"] - run[-1]["t"] != MINUTE):
            if run:
                runs.append(run)
                run = []
        if valid:
            run.append(p)
    if run:
        runs.append(run)
    certificates = []
    for i, run in enumerate(runs):
        cert = infer_run(run, config["curvature_absolute_tolerance"])
        cert["id"] = i
        certificates.append(cert)
        for p in run:
            p["run"] = i
    return points, meals, photos, mapping, certificates


def make_bins(points, certs):
    by_time = defaultdict(list)
    for p in points:
        by_time[p["t"]].append(p)
    known = [c for c in certs if c["phase"] is not None]
    # Grid definitions may be reconstructed retrospectively, but their first
    # evidence and all minute dependencies are checked separately at each origin.
    first_phase = known[0]["phase"] if known else 0
    first_time, last_time = min(by_time), max(by_time)
    epochs = [(first_time, first_phase, "conditional_phase" if known else "unresolved_display_grid")]
    previous_known = None
    for cert in known:
        if cert["phase"] != epochs[-1][1]:
            gap = cert["start"] - previous_known["end"] if previous_known else timedelta(0)
            if gap >= MINUTE * 125:
                epochs.append((cert["start"], cert["phase"], "conditional_phase_after_long_gap"))
            # A change without a long gap is not repaired; phase conflicts below
            # make affected inputs unresolved. It requires source clarification.
        previous_known = cert
    bins = []
    for e, (start, phase, basis) in enumerate(epochs):
        end = epochs[e + 1][0] - MINUTE if e + 1 < len(epochs) else last_time
        t = ceiling(start, phase)
        while t <= end:
            minute_sets = [by_time.get(t + MINUTE * k, []) for k in range(-4, 1)]
            contributors = [p for group in minute_sets for p in group]
            distinct = sum(len(group) == 1 for group in minute_sets)
            complete = distinct == 5 and all(p["v"] is not None for p in contributors)
            run_ids = {p.get("run") for p in contributors}
            cert = certs[next(iter(run_ids))] if complete and len(run_ids) == 1 and None not in run_ids else None
            phase_match = cert is not None and cert["phase"] == phase
            reason = "complete" if complete else ("duplicate_minute" if any(len(g) > 1 for g in minute_sets) else "incomplete_or_invalid_minutes")
            bins.append({"t": t, "phase": phase, "grid_basis": basis, "epoch": e,
                         "v": math.fsum(p["v"] for p in contributors) / 5 if complete else None,
                         "complete": complete, "reason": reason, "contributors": contributors,
                         "distinct": distinct, "cert": cert, "phase_match": phase_match,
                         "dependency": max(ceiling(p["t"], cert["phase"]) for p in contributors) if phase_match else None})
            t += FIVE
    return bins


def segment_bins(bins, config):
    # Long missing runs, including edges, are outside all natural segments.
    i = 0
    while i < len(bins):
        if bins[i]["complete"]:
            i += 1
            continue
        j = i
        while j < len(bins) and not bins[j]["complete"]:
            j += 1
        for b in bins[i:j]:
            b["long_gap"] = j - i >= config["discontinuity_positions"]
            b["missing_run_positions"] = j - i
        i = j
    segment = -1
    last = None
    offset = 0
    for b in bins:
        b["segment"] = None
        b["washout"] = False
        if b.get("long_gap"):
            last = None
            continue
        if last is None or b["t"] - last["t"] != FIVE or b["epoch"] != last["epoch"]:
            segment += 1
            offset = 0
        b["segment"] = segment
        b["washout"] = offset < config["washout_positions"]
        offset += 1
        last = b
    return segment + 1


def causal(bin_, origin):
    cert = bin_["cert"]
    return (bin_["phase_match"] and cert["known_at"] is not None and cert["known_at"] <= origin
            and (cert["invalid_at"] is None or cert["invalid_at"] > origin)
            and bin_["dependency"] is not None and bin_["dependency"] <= origin)


def assign_splits(bins):
    eligible = [b for b in bins if b["complete"] and b["segment"] is not None and not b["washout"] and causal(b, b["t"])]
    n = len(eligible)
    # Exact integer arithmetic avoids floating-point cutoff ambiguity.
    counts = [n * 60 // 100, n * 20 // 100]
    counts.append(n - sum(counts))
    train_end = eligible[counts[0] - 1]["t"] if counts[0] else None
    val_end = eligible[counts[0] + counts[1] - 1]["t"] if counts[1] else train_end
    for b in bins:
        b["partition"] = "train" if train_end is not None and b["t"] <= train_end else ("validation" if counts[1] and b["t"] <= val_end else "test")
        b["split_count_eligible"] = b["complete"] and b["segment"] is not None and not b["washout"] and causal(b, b["t"])
    return {"N": n, "n_train": counts[0], "n_validation": counts[1], "n_test": counts[2],
            "train_last_counted_timestamp": ts(train_end), "validation_last_counted_timestamp": ts(val_end),
            "test_first_counted_timestamp": ts(eligible[sum(counts[:2])]["t"]) if counts[2] else "",
            "boundary_policy": "endpoints_inclusive_all_remaining_grid_positions_follow_timestamp_cutoffs"}


def context_fills(bins, config):
    i = 0
    while i < len(bins):
        b = bins[i]
        if b["v"] is not None or b["segment"] is None or b["washout"]:
            i += 1
            continue
        j = i
        while j < len(bins) and bins[j]["v"] is None and (bins[j]["segment"], bins[j]["partition"]) == (b["segment"], b["partition"]):
            j += 1
        left = bins[i - 1] if i else None
        right = bins[j] if j < len(bins) else None
        def anchor_ok(a):
            return a is not None and a["complete"] and not a["washout"] and (a["segment"], a["partition"]) == (b["segment"], b["partition"])
        if j - i <= config["max_context_gap_positions"] and anchor_ok(left) and anchor_ok(right):
            for k in range(i, j):
                weight = (bins[k]["t"] - left["t"]).total_seconds() / (right["t"] - left["t"]).total_seconds()
                bins[k]["fill"] = {"value": left["v"] + weight * (right["v"] - left["v"]),
                                   "left": i - 1, "right": j, "available_at": right["t"], "count": j - i}
        i = j


def resolve_image(pid, raw, inventory):
    prefix = f"source://CGMacros/CGMacros-{pid}/"
    normalized = raw.strip().replace("\\", "/")
    if normalized.startswith("source://"):
        exact = normalized
    elif normalized.casefold().startswith(f"CGMacros-{pid}/".casefold()):
        exact = "source://CGMacros/" + normalized
    else:
        exact = prefix + normalized.removeprefix("./")
    if ".." in normalized.split("/") or not exact.casefold().startswith((prefix + "photos/").casefold()):
        return "", "invalid_or_out_of_participant_reference"
    if exact in inventory:
        return exact, "exact" if normalized == raw else "recorded_slash_or_whitespace_normalization"
    matches = [uri for uri in inventory if uri.casefold() == exact.casefold()]
    if len(matches) == 1:
        return matches[0], "unique_case_mapping"
    return "", "ambiguous_case_mapping" if matches else "object_not_in_inventory"


def canonical_meals(pid, meals, photos, bins, inventory):
    times = [b["t"] for b in bins]
    image_rows, image_by_line = [], {}
    for p in photos:
        uri, match = resolve_image(pid, p["image_reference"], inventory)
        post = bool(POST.search(p["image_reference"]))
        eligible = bool(p["meal_type"] and uri and not post)
        reason = "typed_row_operational_pre_meal_candidate" if eligible else ("explicit_post_completion_leftover_or_empty_plate" if post else ("photo_only_row" if not p["meal_type"] else match))
        image_row = {"participant": pid, "source_line": p["line"], "original_meal_timestamp": ts(p["t"]),
                     "meal_type_audit_only": p["meal_type"], "original_reference": p["image_reference"], "object_uri": uri,
                     "match_method": match, "inventory_bytes": inventory[uri]["bytes"] if uri else "",
                     "selection_reason": reason, "operational_candidate": int(eligible), "selected_for_acquisition": 0,
                     "image_status": "pending_acquisition" if eligible else "rejected_or_unresolved",
                     "explicit_start_token": int(bool(ROLE.search(p["image_reference"]))), "canonical_meal_id": "",
                     "partition": "", "visual_check_status": "not_checked", "verified_image_available": 0}
        image_rows.append(image_row)
        image_by_line[p["line"]] = image_row
    groups = defaultdict(list)
    for m in meals:
        k = bisect.bisect_left(times, m["t"])
        aligned = times[k] if k < len(times) and times[k] - m["t"] < FIVE else None
        m["grid_index"], m["aligned"] = k if aligned is not None else None, aligned
        groups[aligned if aligned is not None else ("unaligned", m["line"])].append(m)
    canonical, candidates = [], []
    for _, group in sorted(groups.items(), key=lambda pair: min(m["t"] for m in pair[1])):
        def rank(m):
            cal = number(m["nutrients"]["Calories"])
            return (-(cal if cal is not None else -math.inf), m["t"], m["line"])
        winner = min(group, key=rank)
        meal_id = f"CGMacros-{pid}:meal:{winner['line']}"
        b = bins[winner["grid_index"]] if winner["grid_index"] is not None else None
        image = image_by_line.get(winner["line"])
        nutrient_valid = all(number(v) is not None and number(v) >= 0 for v in winner["nutrients"].values())
        record = {"participant": pid, "meal_id": meal_id, "source_line": winner["line"], "original_meal_timestamp": ts(winner["t"]),
                  "aligned_meal_timestamp": ts(winner["aligned"]), "alignment_delay_minutes": (winner["aligned"] - winner["t"]).total_seconds() / 60 if b else "",
                  "grid_phase": b["phase"] if b else "", "natural_segment": b["segment"] if b else "",
                  "partition": b["partition"] if b else "", "meal_type_audit_only": winner["meal_type"],
                  **{n: winner["nutrients"][n] for n in NUTRIENTS}, "nutrients_valid": int(nutrient_valid),
                  "nutrient_adjustment": "none", "image_object_uri": image["object_uri"] if image else "",
                  "image_candidate_selected": int(bool(image and image["operational_candidate"])), "verified_image_available": 0,
                  "group_candidate_count": len(group), "selection_rule": "largest_finite_calories_then_earliest_source_time_then_lowest_source_line",
                  "availability_provenance": "source_timestamp_only_receipt_and_annotation_time_unverified"}
        canonical.append(record)
        for m in group:
            candidates.append({"participant": pid, "source_line": m["line"], "source_timestamp": ts(m["t"]),
                               "aligned_timestamp": ts(m["aligned"]), "meal_type_audit_only": m["meal_type"],
                               **m["nutrients"], "image_reference": m["image_reference"], "canonical_meal_id": meal_id,
                               "selected_canonical_record": int(m is winner), "group_candidate_count": len(group)})
            img = image_by_line.get(m["line"])
            if img:
                img["canonical_meal_id"] = meal_id
                img["partition"] = record["partition"]
                if m is winner and img["operational_candidate"]:
                    img["selected_for_acquisition"] = 1
                elif img["operational_candidate"]:
                    img["selection_reason"] = "nonwinning_same_meal_bin_candidate"
                    img["image_status"] = "not_selected_same_bin"
    return canonical, candidates, image_rows


def origins(pid, bins, meals, config):
    rows = []
    meal_times = [datetime.fromisoformat(m["aligned_meal_timestamp"]) for m in meals if m["aligned_meal_timestamp"]]
    aligned_meals = [m for m in meals if m["aligned_meal_timestamp"]]
    onset = {t: m for t, m in zip(meal_times, aligned_meals)}
    c, h = config["context_positions"], config["target_positions"]
    for i, b in enumerate(bins):
        t = b["t"]
        reasons = set()
        if b["segment"] is None:
            reasons.add("natural_discontinuity")
        if b["washout"]:
            reasons.add("natural_start_washout")
        history = bins[max(0, i - c + 1):i + 1]
        future = bins[i + 1:i + 1 + h]
        if len(history) != c:
            reasons.add("insufficient_history_positions")
        if len(future) != h:
            reasons.add("insufficient_future_positions")
        window = history + future
        if any(x["partition"] != b["partition"] for x in window):
            reasons.add("partition_boundary_crossing")
        if any(x["segment"] is None or x["segment"] != b["segment"] for x in window):
            reasons.add("natural_segment_crossing")
        if any(y["t"] - x["t"] != FIVE for x, y in zip(window, window[1:])):
            reasons.add("grid_phase_or_spacing_change")
        if any(x["washout"] for x in window):
            reasons.add("washout_in_window")
        if any(not x["complete"] for x in future):
            reasons.add("incomplete_source_targets")
        if not b["complete"] or not causal(b, t):
            reasons.add("origin_endpoint_not_conditional_causal_anchor")
        fill_count, dependencies, evidence_times = 0, [], []
        for x in history:
            anchors = [x]
            if not x["complete"]:
                fill = x.get("fill")
                if not fill:
                    reasons.add("unfillable_context_gap")
                    continue
                if fill["available_at"] > t:
                    reasons.add("context_right_anchor_after_origin")
                anchors = [bins[fill["left"]], bins[fill["right"]]]
                fill_count += 1
            for a in anchors:
                if not causal(a, t):
                    reasons.add("unresolved_source_phase_or_native_dependency")
                if a["dependency"]:
                    dependencies.append(a["dependency"])
                if a["cert"] and a["cert"]["known_at"]:
                    evidence_times.append(a["cert"]["known_at"])
        mi = bisect.bisect_right(meal_times, t) - 1
        latest = aligned_meals[mi] if mi >= 0 else None
        elapsed = (t - datetime.fromisoformat(latest["original_meal_timestamp"])).total_seconds() / 60 if latest else None
        recent = latest is not None and elapsed <= config["recent_meal_max_minutes_inclusive"]
        same_group = latest is not None and latest["partition"] == b["partition"] and latest["natural_segment"] == b["segment"]
        active = bool(recent and same_group)
        row = {"origin_id": f"CGMacros-{pid}:{ts(t)}", "participant": pid, "origin_timestamp": ts(t),
               "partition": b["partition"], "natural_segment": b["segment"], "grid_phase": b["phase"],
               "eligible": int(not reasons), "rejection_reasons": "|".join(sorted(reasons)),
               "context_start": ts(t - FIVE * (c - 1)), "target_start": ts(t + FIVE), "target_end": ts(t + FIVE * h),
               "context_positions": c, "target_positions": h, "horizon_prefixes": "6|12|24", "context_pipeline_fill_count": fill_count,
               "max_conditional_native_dependency": ts(max(dependencies)) if dependencies else "",
               "max_phase_evidence_time": ts(max(evidence_times)) if evidence_times else "",
               "input_provenance": "conditional_publisher_linear_interpolation_inference",
               "target_provenance": CONVENTION, "native_observations_verified": 0,
               "meal_onset": int(t in onset), "onset_meal_id": onset[t]["meal_id"] if t in onset else "",
               "last_meal_id": latest["meal_id"] if latest else "", "recent_meal_available": int(active),
               "last_meal_mask_reason": "active_source_timestamp_assumption" if active else ("different_partition_or_segment" if recent else "none_or_older_than_six_hours"),
               "image_available": 0, "image_candidate_available_pending_acquisition": int(active and latest["image_candidate_selected"]),
               "nutrition_available": int(active and latest["nutrients_valid"]),
               "elapsed_hours_capped_or_neutral": min(elapsed / 60, 6) if active else config["neutral_elapsed_hours"]}
        for minutes in (30, 60, 120):
            row[f"new_meal_within_{minutes}_minutes"] = int(bisect.bisect_right(meal_times, t + MINUTE * minutes) > bisect.bisect_right(meal_times, t))
        rows.append(row)
    return rows


def qc_participant(pid, bins, origin_rows, meals, split, config):
    valid = [r for r in origin_rows if r["eligible"]]
    # Independent invariants on the realized output, including anchors not stored
    # as part of a context slice when the slice begins inside a gap.
    by_time = {ts(b["t"]): i for i, b in enumerate(bins)}
    violations = []
    for r in valid:
        i = by_time[r["origin_timestamp"]]
        hist, fut = bins[i - 287:i + 1], bins[i + 1:i + 25]
        if len(hist) != 288 or len(fut) != 24:
            violations.append("length")
        if any(not x["complete"] or x["v"] is None for x in fut):
            violations.append("filled_or_incomplete_target")
        for x in hist + fut:
            if (x["partition"], x["segment"]) != (r["partition"], r["natural_segment"]) or x["washout"]:
                violations.append("isolation")
        for x in hist:
            anchors = [x] if x["complete"] else [bins[x["fill"][key]] for key in ("left", "right")]
            if any(not causal(a, bins[i]["t"]) or (a["partition"], a["segment"]) != (r["partition"], r["natural_segment"]) for a in anchors):
                violations.append("causal_anchor")
    for b in bins:
        if b["complete"] and (len(b["contributors"]) != 5 or [x["t"] for x in b["contributors"]] != [b["t"] + MINUTE * k for k in range(-4, 1)]):
            violations.append("bin_completeness")
    for part, field in (("train", "n_train"), ("validation", "n_validation"), ("test", "n_test")):
        if sum(b["split_count_eligible"] and b["partition"] == part for b in bins) != split[field]:
            violations.append("split_count")
    if any(m["aligned_meal_timestamp"] and datetime.fromisoformat(m["aligned_meal_timestamp"]) < datetime.fromisoformat(m["original_meal_timestamp"]) for m in meals):
        violations.append("meal_backward_alignment")
    if len({r["origin_id"] for r in origin_rows}) != len(origin_rows):
        violations.append("duplicate_origin")
    if violations:
        raise AssertionError(f"QC failed {pid}: {Counter(violations)}")
    reasons = Counter(reason for r in origin_rows for reason in r["rejection_reasons"].split("|") if reason)
    summary = {"participant": pid, "grid_positions": len(bins), "complete_source_bins": sum(b["complete"] for b in bins),
               "missing_bins": sum(not b["complete"] for b in bins), "natural_segments": len({b["segment"] for b in bins if b["segment"] is not None}),
               "discontinuity_positions": sum(b["segment"] is None for b in bins), "washout_positions": sum(b["washout"] for b in bins),
               "washout_complete_bins": sum(b["washout"] and b["complete"] for b in bins),
               "context_fill_positions": sum("fill" in b for b in bins), **{k: split[k] for k in ("N", "n_train", "n_validation", "n_test")},
               "candidate_origins": len(origin_rows), "eligible_origins": len(valid), "canonical_meals": len(meals),
               "qc": "PASS", "rejection_reason_counts_overlapping": json_cell(reasons)}
    summary["complete_bins_lost_to_unresolved_endpoint_phase_after_washout"] = sum(b["complete"] and b["segment"] is not None and not b["washout"] and not causal(b, b["t"]) for b in bins)
    for part in ("train", "validation", "test"):
        summary[f"{part}_origins"] = sum(r["partition"] == part for r in valid)
        summary[f"{part}_meal_onsets"] = sum(r["partition"] == part and r["meal_onset"] for r in valid)
    return summary


def bin_row(pid, b, bins):
    cert = b["cert"]
    fill = b.get("fill")
    return {"participant": pid, "timestamp_local": ts(b["t"]), "grid_phase": b["phase"], "grid_phase_basis": b["grid_basis"],
            "natural_segment": b["segment"], "partition": b["partition"], "washout": int(b["washout"]),
            "source_processed_value": b["v"], "source_bin_complete": int(b["complete"]), "bin_status": b["reason"],
            "distinct_singleton_minutes": b["distinct"], "contributor_count": len(b["contributors"]),
            "contributors": json_cell([{"timestamp": ts(p["t"]), "source_line": p["line"], "published_value": p["raw"], "cleaned_value": p["v"], "numeric_rule": p["rule"]} for p in b["contributors"]]),
            "processing_provenance": CONVENTION, "native_observation_verified": 0,
            "source_run_id": cert["id"] if cert else "", "phase_identified_at": ts(cert["known_at"]) if cert else "",
            "phase_invalid_from": ts(cert["invalid_at"]) if cert else "", "conditional_dependency_time": ts(b["dependency"]),
            "split_count_eligible": int(b["split_count_eligible"]), "context_interpolation_value": fill["value"] if fill else "",
            "context_interpolation_left": ts(bins[fill["left"]]["t"]) if fill else "",
            "context_interpolation_right": ts(bins[fill["right"]]["t"]) if fill else "",
            "context_interpolation_available_at": ts(fill["available_at"]) if fill else "",
            "context_interpolation_gap_positions": fill["count"] if fill else "",
            "context_interpolation_provenance": "pipeline_context_only_origin_gated" if fill else ""}


def loso_splits(bins,pid,held='001'):
    counted=[b for b in bins if b['complete'] and b['segment'] is not None and not b['washout'] and causal(b,b['t'])]
    n=len(counted); nt=0 if pid==held else n*80//100; nv=0 if pid==held else n-nt
    end=counted[nt-1]['t'] if nt else None
    for b in bins:
        b['partition']='test' if pid==held else ('train' if end is not None and b['t']<=end else 'validation')
        b['split_count_eligible']=b['complete'] and b['segment'] is not None and not b['washout'] and causal(b,b['t'])
    return dict(N=n,n_train=nt,n_validation=nv,n_test=n if pid==held else 0,
        train_last_counted_timestamp=ts(end),validation_last_counted_timestamp=ts(counted[-1]['t']) if nv else '',
        test_first_counted_timestamp=ts(counted[0]['t']) if pid==held and n else '',
        validation_first_counted_timestamp=ts(counted[nt]['t']) if nv else '',
        boundary_policy='floor_N_times_80_div_100_train; remainder_validation; inclusive_timestamp_cutoff; held_out_all_test')
