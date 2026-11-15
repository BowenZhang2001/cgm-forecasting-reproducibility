"""Participant-macro multimodal errors, curve outcomes and matched intervals."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from cgm_pipeline.reproduce_benchmark_metrics import BOOTSTRAP_DRAWS, estimates


MODELS = ("M0", "M1", "M2", "M3", "M3-N", "M4", "M5", "M5-N")
SEEDS = (20261014, 20261015, 20261016, 20261017, 20261018)
PAIRS = (
    ("M1", "M0"), ("M2", "M1"), ("M4", "M2"), ("M3", "M2"),
    ("M3-N", "M2"), ("M5", "M4"), ("M5-N", "M4"), ("M3-N", "M3"), ("M5-N", "M5"),
    ("M5", "M3"), ("M5-N", "M3-N"),
)


def fold_data(path):
    targets = np.load(path / "test_target.npy", allow_pickle=False).astype(np.float64)
    origin = pd.read_parquet(path / "test_origins.parquet")
    model_predictions = {"M0": {0: np.load(path / "M0.npy", allow_pickle=False)}}
    model_predictions.update({
        model: {seed: np.load(path / model / str(seed) / "prediction.npy", allow_pickle=False)
                for seed in SEEDS}
        for model in MODELS if model != "M0"
    })
    return origin, targets, model_predictions


def combine_folds(folds):
    return (
        pd.concat([fold[0] for fold in folds], ignore_index=True),
        np.concatenate([fold[1] for fold in folds]),
        {model: {seed: np.concatenate([fold[2][model][seed] for fold in folds])
                 for seed in folds[0][2][model]} for model in MODELS},
    )


def shared_bootstrap(people):
    token = int(hashlib.sha256("|".join(people).encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(np.random.SeedSequence([20261019, token]))
    return rng.integers(0, len(people), size=(BOOTSTRAP_DRAWS, len(people)), dtype=np.int16)


def metric_rows(values, people, key):
    seeds_by_person = {model: np.stack(runs).mean(axis=0) for model, runs in values.items()}
    samples = shared_bootstrap(people)
    rows = []
    for model, person in seeds_by_person.items():
        rows.append(dict(key, model=model, reference=None, n_participants=len(people),
                         **estimates(person.mean(), person[samples].mean(axis=1))))
    for model, reference in PAIRS:
        difference = seeds_by_person[model] - seeds_by_person[reference]
        rows.append(dict(key, model=model, reference=reference, n_participants=len(people),
                         **estimates(difference.mean(), difference[samples].mean(axis=1))))
    return rows


def error_population(analysis, target, data, predictions, name, selected):
    if not np.any(selected):
        return []
    people = sorted(set(data.participant[selected]))
    rows = []
    ids = data.participant.to_numpy()
    for horizon in (6, 12, 24):
        y = target[:, :horizon]
        for metric in ("RMSE", "MAE", "MARD"):
            values = {}
            for model, runs in predictions.items():
                values[model] = []
                for values_by_run in runs.values():
                    residual = values_by_run[:, :horizon].astype(np.float64) - y
                    if metric == "RMSE":
                        measures = residual**2
                    elif metric == "MAE":
                        measures = np.abs(residual)
                    else:
                        measures = 100. * np.abs(residual) / y
                    per_person = np.array([measures[selected & (ids == p)].mean() for p in people])
                    values[model].append(np.sqrt(per_person) if metric == "RMSE" else per_person)
            rows.extend(metric_rows(values, people, dict(
                analysis=analysis, population=name, horizon_minutes=horizon * 5, metric=metric,
                n_origins=int(selected.sum()),
            )))
    return rows


def curve_population(analysis, origins, predictions):
    if "baseline" not in origins:
        return []
    selected = origins.baseline.notna().to_numpy()
    data = origins.loc[selected]
    if data.empty:
        return []
    people = sorted(set(data.participant))
    values = {}
    for model, runs in predictions.items():
        values[model] = {"peak_rise_MAE": [], "positive_iAUC_MAE": []}
        for predictions_by_run in runs.values():
            curve = np.column_stack([data.origin_glucose.to_numpy(), predictions_by_run[selected]])
            above_baseline = curve - data.baseline.to_numpy()[:, None]
            errors = {
                "peak_rise_MAE": np.abs(above_baseline.max(axis=1) - data.observed_peak),
                "positive_iAUC_MAE": np.abs(np.trapezoid(np.maximum(above_baseline, 0), dx=5, axis=1)
                                            - data.observed_iauc),
            }
            for metric, e in errors.items():
                values[model][metric].append(np.array([e[data.participant == p].mean() for p in people]))
    return [row for metric in ("peak_rise_MAE", "positive_iAUC_MAE")
            for row in metric_rows({model: metrics[metric] for model, metrics in values.items()}, people,
                                   dict(analysis=analysis, population="meal_curves", horizon_minutes=120,
                                        metric=metric, n_origins=len(data)))]


def analyze(analysis, dataset, glucose_groups):
    origin, target, predictions = dataset
    if len(origin) != len(target) or origin.origin_id.duplicated().any():
        raise ValueError("Forecast and target origins do not align")
    result = error_population(analysis, target, origin, predictions, "overall", np.ones(len(origin), bool))
    result += error_population(analysis, target, origin, predictions, "meal_onset", origin.meal_onset.to_numpy(bool))
    result += curve_population(analysis, origin, predictions)
    if analysis == "ID":
        for label in ("non-diabetes", "prediabetes", "T2D"):
            mask = origin.participant.map(glucose_groups).eq(label).to_numpy()
            result += error_population(analysis, target, origin, predictions, label, mask)
    return result


def glycemic_groups(cgmacros):
    bio, = list(cgmacros.rglob("bio.csv"))
    data = pd.read_csv(bio)
    return {f"{int(pid):03d}": ("non-diabetes" if a1c < 5.7 else "prediabetes" if a1c < 6.5 else "T2D")
            for pid, a1c in zip(data.subject, data["A1c PDL (Lab)"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--cgmacros", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    glucose_groups = glycemic_groups(args.cgmacros)
    rows = []
    if (args.forecasts / "ID").is_dir():
        rows += analyze("ID", fold_data(args.forecasts / "ID"), glucose_groups)
    if (args.forecasts / "LOSO").is_dir():
        folds = sorted(path for path in (args.forecasts / "LOSO").iterdir() if path.is_dir())
        if len(folds) != 45:
            raise ValueError("Expected 45 held-out participant folds")
        rows += analyze("LOSO", combine_folds([fold_data(path) for path in folds]), glucose_groups)
    if not rows:
        parser.error("Multimodal forecasts missing")
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.loc[frame.reference.isna()].drop(columns="reference").to_csv(
        args.output / "multimodal_errors.csv", index=False)
    frame.loc[frame.reference.notna()].to_csv(args.output / "multimodal_paired_effects.csv", index=False)


if __name__ == "__main__":
    main()
