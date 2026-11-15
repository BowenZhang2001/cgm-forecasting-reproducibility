"""Exact acceptance rules for native AutoARIMA fits."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import struct


class AutoARIMAStop(RuntimeError):
    """Fatal operational/QC error, never convertible to a missing prediction."""


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def sequence(value):
    return value if isinstance(value, list) else ([] if value is None else [value])


def doubles(payload, n):
    try:
        raw = base64.b64decode(payload, validate=True)
        if len(raw) != n * 8:
            raise ValueError("wrong float64 length")
        values = list(struct.unpack("<" + "d" * n, raw))
    except (ValueError, TypeError, struct.error) as exc:
        raise AutoARIMAStop("malformed float64 worker payload") from exc
    return values


IDENTITY_FIELDS = ("scope", "fold", "dataset", "role", "partition", "participant_id",
                   "origin_id", "context_points", "context_sha256", "origin_time",
                   "origin_time_basis", "target_bindings")
LOGICAL_FIELDS = ("scope", "fold", "dataset", "partition", "origin_id", "context_points")


def check_identity(identity):
    if any(k not in identity for k in IDENTITY_FIELDS):
        raise AutoARIMAStop("incomplete eligible identity")
    if (identity["context_points"] not in (48, 144, 288)
            or identity["partition"] not in ("train", "validation", "synthetic",
                                               "internal_test", "external_test")
            or any(not isinstance(identity[k], str) or not identity[k]
                   for k in ("scope", "dataset", "role", "participant_id", "origin_id",
                             "origin_time", "origin_time_basis"))
            or not isinstance(identity["context_sha256"], str)
            or len(identity["context_sha256"]) != 64
            or set(identity["target_bindings"]) != {"6", "12", "24"}
            or any(not identity["target_bindings"][h] for h in ("6", "12", "24"))):
        raise AutoARIMAStop("invalid eligible geometry/identity/target binding")


def classify_result(result, identity, *, accept_native_tolerant_constant=False):
    """Validate the model selected by the native AutoARIMA search.

    A native no-suitable-model exception or an iteration-limit fit yields a
    missing forecast. Unclassified worker exceptions stop evaluation.
    """
    check_identity(identity)
    if (result.get("unit_root_test_failed", False) or any(
            isinstance(w,dict) and w.get("message","").startswith("The chosen unit root test encountered an error")
            for w in result.get("search_warnings",[]))):
        raise AutoARIMAStop("native differencing decision unavailable")
    if identity["partition"] in ("internal_test","external_test") and not isinstance(result.get("unit_root_test_failed"),bool):
        raise AutoARIMAStop("unit-root diagnostic flag missing for a test partition")
    if (result.get("id") != identity["origin_id"] + "|context=" + str(identity["context_points"])
            or result.get("partition") != identity["partition"]
            or result.get("width") != identity["context_points"]
            or result.get("horizon") != 24
            or result.get("context_sha256") != identity["context_sha256"]):
        raise AutoARIMAStop("worker/input identity or geometry mismatch")
    if result.get("type") == "error":
        if (result.get("error_stage") == "fit"
                and result.get("error") == "No suitable ARIMA model found"):
            return dict(status="fit_failed", reason="native_no_suitable_model", forecast=None,
                        selected_order=None, optimizer_code=None, actual_optimizer_codes=[],
                        mean=None, drift=None, warnings=result.get("search_warnings", []))
        raise AutoARIMAStop("unclassified worker exception: " + str(result.get("error")))
    required = ("code", "free_parameters", "order", "coef", "candidates",
                "selected_candidate_matches", "search_warnings", "forecast_b64",
                "forecast_finite", "forecast_start", "forecast_frequency", "sigma2",
                "covariance_finite", "covariance_diagonal_nonnegative", "loglik", "aicc",
                "mathematical_roots_valid", "native_margin_valid", "exact_constant",
                "native_tolerant_constant", "exact_differenced_constant", "differenced_constant",
                "forecast_terminal_state_unchanged", "mean", "drift")
    if result.get("type") != "result" or any(k not in result for k in required):
        raise AutoARIMAStop("incomplete worker result schema")
    order=sequence(result["order"])
    if (len(order) != 3 or any(type(x) is not int for x in order)
            or not (0 <= order[0] <= 5 and 0 <= order[1] <= 2 and 0 <= order[2] <= 5)
            or type(result["code"]) is not int or result["code"] not in (0, 1)
            or type(result["free_parameters"]) is not int or result["free_parameters"] < 0):
        raise AutoARIMAStop("unknown optimizer status/result schema")
    coef = result["coef"]
    if not isinstance(coef, (dict, list)) or any(v is not None and type(v) not in (int,float)
            for v in (coef.values() if isinstance(coef, dict) else coef)):
        raise AutoARIMAStop("coefficient schema changed")
    pred = doubles(result["forecast_b64"], 24)
    for field in ("forecast_finite", "covariance_finite", "covariance_diagonal_nonnegative",
                  "mathematical_roots_valid", "native_margin_valid", "exact_constant",
                  "native_tolerant_constant", "exact_differenced_constant", "differenced_constant"):
        if not isinstance(result[field], bool):
            raise AutoARIMAStop("worker numeric-check flag schema changed")
    if (sequence(result["forecast_start"]) != [identity["context_points"] + 1, 1]
            or result["forecast_frequency"] != 1
            or result["forecast_terminal_state_unchanged"] is not True):
        raise AutoARIMAStop("forecast alignment or terminal-state invariant failed")
    if (result.get("same_fit_prefix_bitwise") is not None
            and not all(x is True for x in sequence(result["same_fit_prefix_bitwise"]))):
        raise AutoARIMAStop("horizon prefix consistency failed")
    selected = [c for c in result["candidates"]
                if c["index"] in sequence(result["selected_candidate_matches"])]
    if not selected or any(c.get("coef") != coef or c.get("loglik") != result["loglik"]
                           or c.get("free_parameters") != result["free_parameters"]
                           or c.get("code") != result["code"] for c in selected):
        raise AutoARIMAStop("selected coefficient/likelihood/status provenance mismatch")
    original = [o for c in selected for o in c.get("optimizations", [])
                if not o.get("zero_iteration_rescore", False)]
    codes = [o["code"] for o in original]
    outcome = dict(status="valid", reason=None, forecast=None,
                   selected_order=result["order"], optimizer_code=result["code"],
                   actual_optimizer_codes=codes, mean=result["mean"], drift=result["drift"],
                   warnings=result["search_warnings"],
                   selected_warnings=[w for c in selected for w in c.get("warnings", [])],
                   candidate_warning_count=sum(len(c.get("warnings", [])) for c in result["candidates"]))

    def failed(reason):
        return dict(outcome, status="fit_failed", reason=reason, forecast=None)

    # Require optimizer provenance for models with free coefficients.
    if result["free_parameters"] and not original:
        raise AutoARIMAStop("selected optimizer provenance missing")
    if result["code"] == 1:
        if not codes or codes[-1] != 1:
            raise AutoARIMAStop("selected status differs from true optimizer status")
        return failed("final_iteration_limit")
    if result["free_parameters"] and (not codes or codes[-1] != 0):
        raise AutoARIMAStop("rescore cannot replace original optimizer termination")
    if not all(math.isfinite(v) for v in pred) or result["forecast_finite"] is not True:
        return failed("final_nonfinite_forecast")
    if not all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in (coef.values() if isinstance(coef, dict) else coef)):
        return failed("final_nonfinite_coefficients")
    if result["mathematical_roots_valid"] is not True or result["native_margin_valid"] is not True:
        return failed("final_invalid_roots")
    if not result["covariance_finite"] or not result["covariance_diagonal_nonnegative"]:
        return failed("final_invalid_covariance")
    exact = result["exact_constant"] is True
    exact_diff = result["exact_differenced_constant"] is True and not exact
    tolerant = (result["native_tolerant_constant"] or result["differenced_constant"]) and not (exact or exact_diff)
    if tolerant and not accept_native_tolerant_constant:
        raise AutoARIMAStop("native tolerant-constant forecast disabled by configuration")
    if tolerant and result["free_parameters"] != 0:
        return failed("invalid_native_tolerant_constant_branch")
    if exact:
        if (result["order"] != [0, 0, 0] or result["free_parameters"] != 0
                or result["sigma2"] != 0 or not result["mean"]
                or len(coef) != 1 or not all(v == next(iter(coef.values())) for v in pred)):
            return failed("invalid_native_exact_constant_branch")
        outcome["classification"] = "native_exact_constant_analytic"
    elif exact_diff and (result["loglik"] is None or result["aicc"] is None):
        if result["free_parameters"] != 0 or result["order"][1] not in (1, 2):
            return failed("invalid_native_exact_polynomial_branch")
        outcome["classification"] = "native_exact_differenced_constant_analytic"
    else:
        for k in ("loglik", "aicc", "sigma2"):
            if not isinstance(result[k], (float, int)) or not math.isfinite(result[k]):
                return failed("final_invalid_likelihood_or_variance")
        if result["sigma2"] <= 0:
            return failed("final_invalid_likelihood_or_variance")
        outcome["classification"] = ("native_tolerant_constant" if tolerant else
                                     "no_free_coefficients" if not result["free_parameters"] else
                                     "optimizer_success")
    outcome["forecast"] = pred
    return outcome
