"""Fit one AutoARIMA model per origin and context, reusing its horizon prefixes."""

import argparse
import base64
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

from cgm_pipeline.autoarima_execution import classify_result
from cgm_pipeline.benchmark_inputs import SCOPES, test_windows
from cgm_pipeline.reproduce_baselines import CONFIG, _prediction_schema, write_predictions
from cgm_pipeline.training_data import CONTEXTS, HORIZONS


def send(worker, request):
    worker.stdin.write(json.dumps(request) + "\n")
    worker.stdin.flush()
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError("R forecast worker exited before returning a fit")
    return json.loads(line)


def forecast(worker, scope, row, context):
    packed = np.asarray(context, dtype="<f8").tobytes()
    sha = hashlib.sha256(packed).hexdigest()
    request = {
        "partition": row["partition"], "width": len(context), "horizon": 24,
        "id": f"{row['origin_id']}|context={len(context)}",
        "context_b64": base64.b64encode(packed).decode(),
        "context_sha256": sha,
    }
    result = send(worker, request)
    surrogate_time = row["origin_id"].rsplit("@", 1)[-1]
    identity = {
        "scope": scope, "fold": row["fold"], "dataset": row["dataset"],
        "partition": row["partition"], "role": row["partition"],
        "participant_id": row["participant_id"], "origin_id": row["origin_id"],
        "origin_time": surrogate_time, "origin_time_basis": "saved forecast origin",
        "context_sha256": sha, "context_points": len(context),
        "target_bindings": {str(h): f"{row['origin_id']}|horizon={h}" for h in HORIZONS.values()},
    }
    return classify_result(result, identity, accept_native_tolerant_constant=True)["forecast"]


def run_scope(worker, work, output, scope_name):
    scope = SCOPES[scope_name]
    schemas = {horizon: _prediction_schema(horizon) for horizon in HORIZONS}
    for hours in CONTEXTS:
        outputs = {}
        for minutes in HORIZONS:
            destination = output / scope_name / f"{hours}h-{minutes}m/AutoARIMA/deterministic"
            destination.mkdir(parents=True, exist_ok=False)
            outputs[minutes] = destination
        for dataset, partition, batches in test_windows(work, scope, hours, 120):
            with ExitStack() as stack:
                writers = {
                    m: stack.enter_context(pq.ParquetWriter(
                        path / f"{partition}.{dataset}.parquet", schemas[m],
                    ))
                    for m, path in outputs.items()
                }
                for batch in batches:
                    rows = batch.metadata.to_dict("records")
                    predictions = [forecast(worker, scope_name, row, x) for row, x in zip(rows, batch.context)]
                    for minutes, horizon in HORIZONS.items():
                        values = [None if item is None else item[:horizon] for item in predictions]
                        write_predictions(writers[minutes], schemas[minutes], batch, values, horizon)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r-library", type=Path, required=True)
    parser.add_argument("--scope", nargs="+", choices=tuple(SCOPES), default=tuple(SCOPES))
    args = parser.parse_args()
    config_path = CONFIG / "autoarima.json"
    config = json.loads(config_path.read_text())
    environment = dict(os.environ, **config["worker_environment"], AUTOARIMA_R_OBSERVER="1")
    with subprocess.Popen([
        "Rscript", "--vanilla", str(CONFIG.parent / "scripts/forecasting/autoarima_r_worker.R"),
        str(args.r_library.resolve()), str(config_path),
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=environment) as worker:
        try:
            if json.loads(worker.stdout.readline())["type"] != "ready":
                raise RuntimeError("R forecast worker did not initialize")
            for scope in args.scope:
                run_scope(worker, args.work_root, args.output, scope)
        finally:
            worker.stdin.close()
        if worker.wait() != 0:
            raise RuntimeError(f"R forecast worker exited with code {worker.returncode}")


if __name__ == "__main__":
    main()
