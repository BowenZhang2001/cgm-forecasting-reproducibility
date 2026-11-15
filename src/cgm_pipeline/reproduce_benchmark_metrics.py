"""Aggregate benchmark errors, clinical rates and paired percentile intervals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cgm_pipeline.benchmark_inputs import PRIMARY_EXTERNAL, SCOPES
from cgm_pipeline.clinical_metrics import (
    _dataset_bootstrap_seed, _has_three_consecutive, clarke_zone_codes,
)
from cgm_pipeline.reproduce_baselines import CONFIG


MODELS = ("LOCF", "AutoARIMA", "ElasticNet", "LSTM", "PatchTST", "CGM-JEPA", "Compact-GluFormer")
TSFMS = ("chronos_bolt_tiny", "chronos_bolt_small", "chronos2_small", "timesfm_2_5_rank8")
BENCHMARK_MODELS = (*MODELS, *(f"{base}-{mode}" for base in TSFMS for mode in ("zs", "adapted")))
BASELINES = ("LOCF", "ElasticNet", "LSTM", "PatchTST")
BOOTSTRAP_DRAWS = 10000
ERROR_METRICS = tuple(f"{region}_{metric}"
                      for region in ("overall", "low", "target", "high")
                      for metric in ("rmse", "mae", "mard"))


def divide(numerator, denominator):
    return np.divide(numerator, denominator,
                     out=np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan),
                     where=denominator > 0)


def nanmean(values, axis=0):
    return divide(np.nansum(values, axis=axis), np.isfinite(values).sum(axis=axis))


def confusion(truth, prediction, eligible):
    return np.stack([
        eligible & truth & prediction, eligible & ~truth & prediction,
        eligible & ~truth & ~prediction, eligible & truth & ~prediction,
    ], axis=1)


def participant_statistics(path):
    data = pq.read_table(path).to_pandas()
    valid = data.prediction.notna().to_numpy()
    truth = np.stack(data.target)
    forecast = np.zeros_like(truth)
    if valid.any():
        forecast[valid] = np.stack(data.prediction[valid])
    residual = forecast - truth
    absolute = np.abs(residual)
    metrics = {}
    masks = {"overall": np.ones_like(truth, dtype=bool), "low": truth < 70,
             "target": (truth >= 70) & (truth <= 180), "high": truth > 180}
    for region, elements in masks.items():
        elements &= valid[:, None]
        denominator = elements.sum(axis=1)
        metrics[f"{region}_rmse"] = np.sqrt(divide(np.where(elements, residual**2, 0).sum(1), denominator))
        metrics[f"{region}_mae"] = divide(np.where(elements, absolute, 0).sum(1), denominator)
        metrics[f"{region}_mard"] = divide(np.where(elements, absolute / truth * 100, 0).sum(1), denominator)
    zones = clarke_zone_codes(truth, forecast)
    zone_counts = {f"zone_{z}": ((zones == i) & valid[:, None]).sum(1) for i, z in enumerate("ABCDE")}
    warning_counts = {}
    for label, actual, predicted, eligible in (
        ("low", truth < 70, forecast < 70, data.origin_glucose.to_numpy() >= 70),
        ("high", truth > 180, forecast > 180, data.origin_glucose.to_numpy() <= 180),
    ):
        counts = confusion(_has_three_consecutive(actual), _has_three_consecutive(predicted), eligible & valid)
        warning_counts.update({f"{label}_{key}": counts[:, i] for i, key in enumerate(("tp", "fp", "tn", "fn"))})
    keys = ["dataset", "participant_id", "disease_subgroup"]
    errors = data[keys].assign(**metrics).groupby(keys, dropna=False).mean()
    counts = data[keys].assign(**zone_counts, **warning_counts).groupby(keys, dropna=False).sum()
    scope, task, model, run = path.parts[-5:-1]
    partition = path.name.split(".", 1)[0]
    return errors.join(counts).reset_index().assign(scope=scope, task=task, model=model, run=run, partition=partition)


def estimates(point, draws):
    draws = np.asarray(draws)
    finite = draws[np.isfinite(draws)]
    lower, upper = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
    return {"estimate": point, "ci_low": lower, "ci_high": upper, "defined_replicates": len(finite)}


def statistic(weights, values):
    return divide(weights @ np.nan_to_num(values, nan=0.), weights @ np.isfinite(values))


def bootstrap_weights(n, seed):
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1. / n), size=BOOTSTRAP_DRAWS)


def errors_by_group(frame, metadata, weights):
    results, rmse = [], {}
    n = weights.shape[1]
    for model, subset in frame.groupby("model", sort=True):
        values = subset.groupby("index")[list(ERROR_METRICS)].mean().reindex(range(n)).to_numpy()
        means = nanmean(values)
        draws = statistic(weights, values)
        rmse[model] = means[0], draws[:, 0]
        results += [dict(metadata, model=model, metric=metric,
                         n_participants=np.isfinite(values[:, i]).sum(),
                         **estimates(means[i], draws[:, i]))
                    for i, metric in enumerate(ERROR_METRICS)]
    return results, rmse


def paired_results(rmse, metadata):
    comparator = min((m for m in BASELINES if m in rmse), key=lambda m: rmse[m][0], default=None)
    rows = []
    for name in TSFMS:
        model = f"{name}-adapted"
        if model not in rmse:
            continue
        references = [("adapted_vs_zs", f"{name}-zs")]
        references += [("adapted_vs_baseline", baseline) for baseline in BASELINES]
        for family, reference in references:
            if reference not in rmse:
                continue
            point, draws = rmse[model]
            ref_point, ref_draws = rmse[reference]
            difference = draws - ref_draws
            finite = difference[np.isfinite(difference)]
            p_value = min(1., 2 * min((np.count_nonzero(finite <= 0) + 1) / (len(finite) + 1),
                                      (np.count_nonzero(finite >= 0) + 1) / (len(finite) + 1)))
            rows.append(dict(metadata, model=model, reference=reference, contrast=family,
                             selected_for_primary=family == "adapted_vs_zs" or reference == comparator,
                             p_value=p_value, **estimates(point - ref_point, difference)))
    return rows


def add_bh(frame):
    frame["q_value"] = np.nan
    for _, group in frame.groupby(["model", "reference", "contrast"]):
        ordered = group.sort_values("p_value")
        q = ordered.p_value.to_numpy() * len(ordered) / np.arange(1, len(ordered) + 1)
        frame.loc[ordered.index, "q_value"] = np.minimum(1., np.minimum.accumulate(q[::-1])[::-1])
    return frame


def numeric_clinical(counts):
    metrics = {}
    zone_counts = np.stack([counts[f"zone_{zone}"] for zone in "ABCDE"])
    total = zone_counts.sum(axis=0)
    metrics.update({f"Clarke_{z}": divide(zone_counts[i], total) for i, z in enumerate("ABCDE")})
    metrics["Clarke_AB"] = divide(zone_counts[0] + zone_counts[1], total)
    for event in ("low", "high"):
        tp, fp, tn, fn = (counts[f"{event}_{c}"] for c in ("tp", "fp", "tn", "fn"))
        metrics.update({
            f"{event}_sensitivity": divide(tp, tp + fn), f"{event}_specificity": divide(tn, tn + fp),
            f"{event}_PPV": divide(tp, tp + fp), f"{event}_F1": divide(2 * tp, 2 * tp + fp + fn),
        })
    return metrics


def clinical_results(frame, metadata, weights):
    rows = []
    keys = [f"zone_{z}" for z in "ABCDE"] + [f"{event}_{c}" for event in ("low", "high")
                                            for c in ("tp", "fp", "tn", "fn")]
    for model, subset in frame.groupby("model", sort=True):
        points, draws = [], []
        for _, run in subset.groupby("run", sort=True):
            matrix = run.set_index("index")[keys].reindex(range(weights.shape[1]), fill_value=0).to_numpy()
            points.append(numeric_clinical(dict(zip(keys, matrix.sum(axis=0)))))
            draws.append(numeric_clinical(dict(zip(keys, (weights @ matrix).T))))
        for name in points[0]:
            rows.append(dict(metadata, model=model, metric=name,
                             **estimates(nanmean([p[name] for p in points]),
                                         nanmean([d[name] for d in draws]))))
    return rows


def add_participant_index(frame, order):
    people = {key: index for index, key in enumerate(order)}
    return frame.assign(index=[people[key] for key in zip(frame.dataset, frame.participant_id)])


def dataset_results(frame):
    errors, paired, clinical = [], [], []
    for (scope, partition, dataset), selection in frame.groupby(["scope", "partition", "dataset"]):
        role = "external" if partition == "external_test" else "internal"
        spec = SCOPES[scope]
        seed = _dataset_bootstrap_seed(20261014, (scope, spec.cohort, role, dataset, spec.fold, partition))
        participants = [(dataset, pid) for pid in sorted(selection.participant_id.unique())]
        weights = bootstrap_weights(len(participants), seed).astype(np.float64)
        selection = add_participant_index(selection, participants)
        for task, part in selection.groupby("task"):
            key = dict(scope=scope, partition=partition, dataset=dataset, task=task)
            rows, rmse = errors_by_group(part, key, weights)
            errors += rows
            paired += paired_results(rmse, key)
            clinical += clinical_results(part, key, weights)
    return errors, paired, clinical


def pooled_results(frame):
    config = json.loads((CONFIG / "clinical_bootstrap.json").read_text())
    results = []
    external = frame.loc[frame.partition == "external_test"]
    for group, strata in config.items():
        selection = external.loc[external.dataset.isin(strata)]
        if selection.empty:
            continue
        participants = sorted(set(zip(selection.dataset, selection.participant_id)),
                              key=lambda p: (p[0], hashlib.sha256("\x1f".join(p).encode()).hexdigest()))
        selection = add_participant_index(selection, participants)
        weights = np.zeros((BOOTSTRAP_DRAWS, len(participants)), dtype=np.float64)
        for dataset, seed in strata.items():
            subset = [i for i, (d, _) in enumerate(participants) if d == dataset]
            if subset:
                weights[:, subset] = bootstrap_weights(len(subset), seed)
        for task, part in selection.groupby("task"):
            results += clinical_results(part, dict(group=group, task=task), weights)
    return results


def subgroups(frame):
    selection = frame.loc[(frame.scope == "primary") & (frame.task == "24h-120m")].copy()
    group_labels = {"prediabetes": "prediabetes", "T2D": "T2D",
                    "non_insulin_medication_controlled": "T2D", "insulin_dependent": "T2D"}
    selection["group"] = selection.disease_subgroup.map(group_labels)
    order = ("AI_READI_2025", *PRIMARY_EXTERNAL)
    records = []
    for ei, (code, partition) in enumerate((("ID", "internal_test"), ("OOD", "external_test"))):
        for gi, group in enumerate(("prediabetes", "T2D")):
            subset = selection.loc[(selection.partition == partition) & (selection.group == group)]
            if subset.empty:
                continue
            averages = subset.groupby(["dataset", "participant_id", "model"]).overall_rmse.mean().unstack()
            index = sorted(averages.index, key=lambda key: (order.index(key[0]), key[1]))
            averages = averages.reindex(index)
            rng = np.random.default_rng(np.random.SeedSequence([20261107, ei, gi]))
            strata = [np.array([i for i, (d, _) in enumerate(index) if d == dataset]) for dataset in order]
            indices = np.concatenate([ix[rng.integers(0, len(ix), (BOOTSTRAP_DRAWS, len(ix)), dtype=np.int32)]
                                      for ix in strata if len(ix)], axis=1)
            for model, estimates_by_person in averages.items():
                values = estimates_by_person.to_numpy()
                summary = estimates(nanmean(values), nanmean(values[indices], axis=1))
                records.append(dict(evaluation_set=code, group=group, model=model,
                                    n_participants=np.isfinite(values).sum(), rmse=summary["estimate"],
                                    ci_low=summary["ci_low"], ci_high=summary["ci_high"]))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.forecasts.glob("*/*/*/*/*_test.*.parquet"))
    if not files:
        parser.error("Forecast directory has no benchmark files")
    frame = pd.concat([participant_statistics(path) for path in files], ignore_index=True)
    if set(frame.model.unique()) != set(BENCHMARK_MODELS):
        parser.error("Complete analysis requires the forecasts from all fifteen models")
    if frame.duplicated(["scope", "task", "dataset", "partition", "model", "run", "participant_id"]).any():
        raise ValueError("Duplicate participant/run cells")
    args.output.mkdir(parents=True, exist_ok=False)
    errors, paired, clinical = dataset_results(frame)
    pd.DataFrame(errors).to_csv(args.output / "benchmark_errors.csv", index=False)
    add_bh(pd.DataFrame(paired)).to_csv(args.output / "benchmark_paired_rmse.csv", index=False)
    pd.DataFrame(clinical).to_csv(args.output / "benchmark_clinical.csv", index=False)
    pd.DataFrame(pooled_results(frame)).to_csv(args.output / "benchmark_pooled_ood_clinical.csv", index=False)
    pd.DataFrame(subgroups(frame)).to_csv(args.output / "benchmark_glycemic_groups.csv", index=False)


if __name__ == "__main__":
    main()
