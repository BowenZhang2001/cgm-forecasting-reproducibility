"""Run the study in order using the separate training environments."""

import argparse
import json
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parent


def path_config(config):
    raw = json.loads(config.read_text())

    def configured_path(item):
        return config.parent / Path(item).expanduser()

    return {
        "python": {
            name: configured_path(path).absolute()
            for name, path in raw.pop("python").items()
        },
        **{name: configured_path(path).resolve() for name, path in raw.items()},
    }


def run(name, command, config):
    python = config["python"][name]
    print(f"[{name}] {command[0]} {command[1]}", flush=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    subprocess.run([python, *command], cwd=REPO, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--stage", choices=("data", "benchmark", "image", "multimodal", "metrics", "all"),
                        default="all")
    args = parser.parse_args()
    conf = path_config(args.paths.resolve())
    work, forecasts = conf["work_root"], conf["work_root"] / "benchmark_forecasts"
    image_output = work / "image_models"
    multimodal = work / "multimodal_forecasts"
    cache = work / "public_checkpoints"
    metric_root = work / "metrics"
    sources = conf["source_paths"]
    cgmacros = conf["cgmacros"]
    common = ["--work-root", work, "--output", forecasts]

    if args.stage in ("data", "all"):
        run("data", ["-m", "cgm_pipeline.reproduce", "--sources", sources, "--work-root", work], conf)
    if args.stage in ("benchmark", "all"):
        run("benchmark", ["-m", "cgm_pipeline.reproduce_baselines", *common], conf)
        run("benchmark", ["-m", "cgm_pipeline.reproduce_autoarima", *common,
                          "--r-library", conf["r_library"]], conf)
        for model, environment in (
            ("chronos_bolt_tiny", "bolt"), ("chronos_bolt_small", "bolt"),
            ("chronos2_small", "chronos2"), ("timesfm_2_5_rank8", "timesfm"),
        ):
            run(environment, ["-m", "cgm_pipeline.reproduce_tsfms", *common,
                              "--checkpoint-cache", cache, "--model", model], conf)
        run("bolt", ["-m", "cgm_pipeline.reproduce_cgm_models", *common, "--checkpoint-cache", cache], conf)
    if args.stage in ("image", "all"):
        run("image", ["-B", "image/scripts/train_encoders.py",
                      "--food101", conf["food101"], "--nutrition5k", conf["nutrition5k"],
                      "--out", image_output], conf)
        run("image", ["-B", "image/scripts/encode_meal_photos.py",
                      "--cgmacros", cgmacros, "--encoder-root", image_output], conf)
    if args.stage in ("multimodal", "all"):
        image_qc = ["--image-qc", conf["image_qc"]] if "image_qc" in conf else []
        run("bolt", ["-m", "cgm_pipeline.reproduce_multimodal", "--work-root", work,
                     "--cgmacros", cgmacros, *image_qc,
                     "--image-encoder-root", image_output, "--checkpoint-cache", cache,
                     "--output", multimodal], conf)
    if args.stage in ("metrics", "all"):
        run("benchmark", ["-m", "cgm_pipeline.check_metric_inputs", "--benchmark", forecasts,
                          "--multimodal", multimodal, "--cgmacros", cgmacros], conf)
        run("benchmark", ["-m", "cgm_pipeline.reproduce_benchmark_metrics",
                          "--forecasts", forecasts, "--output", metric_root / "benchmark"], conf)
        run("benchmark", ["-m", "cgm_pipeline.reproduce_multimodal_metrics",
                          "--forecasts", multimodal, "--cgmacros", cgmacros,
                          "--output", metric_root / "multimodal"], conf)


if __name__ == "__main__":
    main()
