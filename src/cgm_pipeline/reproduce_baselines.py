"""Fit the selected baseline configurations and generate held-out forecasts."""

import argparse
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cgm_pipeline.benchmark_inputs import MODEL_SEEDS, SCOPES, load_scope, test_windows
from cgm_pipeline.elastic_net import fit_frozen_selected_candidate
from cgm_pipeline.formal_training import EarlyStopping, fit_neural_epochs
from cgm_pipeline.locf import predict_locf
from cgm_pipeline.training_data import CONTEXTS, HORIZONS


CONFIG = Path(__file__).resolve().parents[2] / "configs"


def configs(filename, keys):
    with (CONFIG / filename).open(newline="") as stream:
        return {tuple(row[key] for key in keys): row for row in csv.DictReader(stream)}


def train_neural(model, lr, seed, hours, minutes, train, validation, norm, output, device):
    import torch
    from cgm_pipeline.lstm import LSTMTask
    from cgm_pipeline.neural_training import NeuralSettings
    from cgm_pipeline.patchtst import PatchTSTTask

    settings = NeuralSettings(
        seed=seed, learning_rate=lr, weight_decay=1e-5, micro_batch_size=64,
        accumulation_steps=1, deterministic=False,
    )
    task_class = {"LSTM": LSTMTask, "PatchTST": PatchTSTTask}[model]
    task = task_class(
        normalization=norm, context_hours=hours, horizon_minutes=minutes,
        settings=settings, device=device, provenance={"checkpoint_role": "best_validation"},
    )

    def validation_batches():
        for donor in validation:
            x, y = donor.task(hours, minutes)
            for start in range(0, len(x), 64):
                end = start + 64
                yield x[start:end], y[start:end], donor.rows[start:end], "mg/dL"

    def save_best(trainer, epoch):
        nonlocal checkpoint
        if epoch["decision"]["save_best"]:
            checkpoint = io.BytesIO()
            trainer.save_checkpoint(checkpoint)

    x, y = train.task(hours, minutes)
    checkpoint = None
    epochs = fit_neural_epochs(
        task, x, y, origin_rows=train.rows,
        validation_batches=validation_batches,
        validation_index=[row for donor in validation for row in donor.rows],
        expected_training_index=train.rows,
        stop_policy=EarlyStopping(max_epochs=100, patience=5, min_delta_mgdl=0.0),
        partial_policy="actual", on_epoch=save_best,
    )
    if checkpoint is None:
        raise RuntimeError("No eligible validation checkpoint")
    (output / "best.pt").write_bytes(checkpoint.getbuffer())
    (output / "epochs.json").write_text(json.dumps(
        [epoch["decision"] for epoch in epochs], indent=2, allow_nan=False) + "\n")
    checkpoint.seek(0)
    task.restore_state(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    return lambda values, rows: task.predict(values, origin_rows=rows, batch_size=512).predictions


def export_predictions(work, output, scope, hours, minutes, predict, *, batch_size=256):
    for dataset, partition, batches in test_windows(work, scope, hours, minutes, batch_size=batch_size):
        path = output / f"{partition}.{dataset}.parquet"
        schema = _prediction_schema(minutes)
        with pq.ParquetWriter(path, schema) as writer:
            for batch in batches:
                rows = batch.metadata.to_dict("records")
                predictions = predict(batch.context, rows)
                write_predictions(writer, schema, batch, predictions, HORIZONS[minutes])


def write_predictions(writer, schema, batch, predictions, horizon):
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    table = pa.Table.from_pydict({
        "origin_id": batch.metadata.origin_id,
        "participant_id": batch.metadata.participant_id,
        "dataset": batch.metadata.dataset,
        "partition": batch.metadata.partition,
        "disease_subgroup": batch.metadata.disease_subgroup,
        "origin_glucose": batch.context[:, -1].tolist(),
        "target": batch.target[:, :horizon].tolist(),
        "prediction": predictions,
    }, schema=schema)
    writer.write_table(table)


def _prediction_schema(minutes):
    if minutes not in HORIZONS:
        raise ValueError("Unknown forecast horizon")
    trajectory = pa.list_(pa.float64())
    return pa.schema([
        ("origin_id", pa.string()), ("participant_id", pa.string()),
        ("dataset", pa.string()), ("partition", pa.string()), ("disease_subgroup", pa.string()),
        ("origin_glucose", pa.float64()), ("target", trajectory), ("prediction", trajectory),
    ])


def run_scope(work, output, scope_name, tasks, models, device):
    scope = SCOPES[scope_name]
    train, validation, norm, sample = load_scope(work, scope)
    (output / "sample.json").write_text(json.dumps(sample, indent=2) + "\n")
    (output / "normalization.json").write_text(json.dumps(asdict(norm), indent=2) + "\n")
    neural_lr = configs("neural_baseline_learning_rates.csv", ("scope", "model", "task"))
    elastic = configs("elastic_net.csv", ("scope", "task"))
    for hours, minutes in tasks:
        task_id = f"{hours}h-{minutes}m"
        for model in models:
            for seed in MODEL_SEEDS if model in ("LSTM", "PatchTST") else ("deterministic",):
                destination = output / task_id / model / str(seed)
                destination.mkdir(parents=True, exist_ok=False)
                if model == "LOCF":
                    predict = lambda values, _: predict_locf(values, horizon_minutes=minutes)
                elif model == "ElasticNet":
                    spec = elastic[scope_name, task_id]
                    x, y = train.task(hours, minutes)
                    estimator = fit_frozen_selected_candidate(
                        x, y, origin_rows=train.rows, normalization=norm,
                        context_hours=hours, horizon_minutes=minutes,
                        mixing=float(spec["mixing_fraction"]), lambda_value=float(spec["lambda_value"]),
                        max_iter=int(spec["max_iter"]), tol=1e-7,
                    )
                    np.savez_compressed(destination / "estimator.npz",
                                        coefficients=estimator.estimator_.coef_,
                                        intercept=estimator.estimator_.intercept_)
                    predict = lambda values, rows: estimator.predict(values, origin_rows=rows).predictions
                else:
                    lr = float(neural_lr[scope_name, model, task_id]["learning_rate"])
                    predict = train_neural(model, lr, seed, hours, minutes, train, validation,
                                           norm, destination, device)
                export_predictions(work, destination, scope, hours, minutes, predict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scope", nargs="+", choices=tuple(SCOPES), default=tuple(SCOPES))
    parser.add_argument("--models", nargs="+", choices=("LOCF", "ElasticNet", "LSTM", "PatchTST"),
                        default=("LOCF", "ElasticNet", "LSTM", "PatchTST"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tasks = [(hours, minutes) for hours in CONTEXTS for minutes in HORIZONS]
    for name in args.scope:
        output = args.output / name
        output.mkdir(parents=True, exist_ok=False)
        run_scope(args.work_root, output, name, tasks, args.models, args.device)


if __name__ == "__main__":
    main()
