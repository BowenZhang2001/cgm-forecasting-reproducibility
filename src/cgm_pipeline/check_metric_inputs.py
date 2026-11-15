"""Check forecast file coverage before aggregating the complete study."""

import argparse
from itertools import product
from pathlib import Path

from cgm_pipeline.benchmark_inputs import CONTEXTS, FOLDS, HORIZONS, MODEL_SEEDS, PRIMARY_EXTERNAL, SCOPES
from cgm_pipeline.reproduce_benchmark_metrics import BENCHMARK_MODELS
from cgm_pipeline.reproduce_multimodal_metrics import MODELS, SEEDS, glycemic_groups


def benchmark_files(root):
    expected = set()
    for scope, settings in SCOPES.items():
        external = PRIMARY_EXTERNAL if settings.fold is None else (FOLDS[settings.fold][0],)
        filenames = [f"internal_test.{dataset}.parquet" for dataset in settings.development_datasets]
        filenames.extend(f"external_test.{dataset}.parquet" for dataset in external)
        for hours, minutes in product(CONTEXTS, HORIZONS):
            task = root / scope / f"{hours}h-{minutes}m"
            for model in BENCHMARK_MODELS:
                single_run = model in ("LOCF", "AutoARIMA", "ElasticNet") or model.endswith("-zs")
                runs = ("deterministic",) if single_run else tuple(map(str, MODEL_SEEDS))
                expected.update(task / model / run / name for run, name in product(runs, filenames))
    return expected


def multimodal_files(root, participants):
    folds = {root / "LOSO" / person for person in participants}
    filenames = [Path("test_origins.parquet"), Path("test_target.npy"), Path("M0.npy")]
    filenames.extend(Path(model) / str(seed) / "prediction.npy"
                     for model, seed in product(MODELS, SEEDS) if model != "M0")
    expected = {folder / name for folder, name in product(folds | {root / "ID"}, filenames)}
    return expected, folds


def missing_files(expected):
    return {path for path in expected if not path.is_file() or path.stat().st_size == 0}


def describe(label, root, paths):
    examples = "\n".join(f"  {path.relative_to(root)}" for path in sorted(paths)[:5])
    remaining = f"\n  ... and {len(paths) - 5} more" if len(paths) > 5 else ""
    return f"{len(paths)} {label} under {root}\n{examples}{remaining}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--multimodal", required=True, type=Path)
    parser.add_argument("--cgmacros", required=True, type=Path)
    args = parser.parse_args()

    participants = glycemic_groups(args.cgmacros)
    if len(participants) != 45:
        parser.error(f"Expected 45 CGMacros participants in bio.csv, found {len(participants)}")

    benchmark = benchmark_files(args.benchmark)
    multimodal, folds = multimodal_files(args.multimodal, participants)
    actual_benchmark = set(args.benchmark.glob("*/*/*/*/*_test.*.parquet"))
    actual_folds = {path for path in (args.multimodal / "LOSO").glob("*") if path.is_dir()}
    checks = (
        ("missing or empty benchmark forecast files", args.benchmark, missing_files(benchmark)),
        ("unexpected benchmark forecast entries", args.benchmark, actual_benchmark - benchmark),
        ("missing or empty multimodal input files", args.multimodal, missing_files(multimodal)),
        ("unexpected LOSO fold directories", args.multimodal, actual_folds - folds),
    )
    failures = [describe(label, root, paths) for label, root, paths in checks if paths]
    if failures:
        parser.error("\n" + "\n".join(failures))

    print("Forecast file coverage is complete for the benchmark, multimodal ID, and LOSO.")


if __name__ == "__main__":
    main()
