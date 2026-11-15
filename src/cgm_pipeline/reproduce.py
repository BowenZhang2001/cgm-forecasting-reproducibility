"""Build the study's benchmark datasets from locally acquired source files."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from cgm_pipeline.build import ai_readi, broll, cgmacros, hall, t1d, t1d_folds
from cgm_pipeline.ingestion import broll as broll_ingestion


REPO = Path(__file__).resolve().parents[2]
RAW_FOLDERS = (*t1d.DATASETS, ai_readi.DATASET, hall.DATASET, cgmacros.DATASET)


def source_paths(config_path):
    config = json.loads(config_path.read_text())
    expected = {*RAW_FOLDERS, "Broll_2021"}
    if set(config) != expected:
        raise ValueError(f"Raw source settings must list {sorted(expected)}")
    paths = {dataset: (config_path.parent / value).expanduser().resolve()
             if not str(value).startswith("~") else Path(value).expanduser().resolve()
             for dataset, value in config.items()}
    for dataset, path in paths.items():
        exists = path.is_file() if dataset == "Broll_2021" else path.is_dir()
        if not exists:
            raise FileNotFoundError(f"{dataset} source not found at {path}")
    return paths


def link_source(work, dataset, source):
    leaf = dataset if dataset == hall.DATASET else "source"
    link = work / "acquisition" / dataset / leaf
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == source:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Source slot already in use: {link}")
    link.symlink_to(source, target_is_directory=True)


def organize_cgmacros(work, source):
    """Expose original CSV bytes from the PhysioNet archive in the builder layout."""
    folder = work / "source_layout/CGMacros"
    folder.mkdir(parents=True, exist_ok=True)
    labels = list(source.rglob("bio.csv"))
    tables = sorted(source.rglob("CGMacros-[0-9][0-9][0-9].csv"))
    if len(labels) != 1 or len(tables) != 45 or len({path.name for path in tables}) != 45:
        raise ValueError("Expected the original CGMacros bio.csv and 45 unique participant CSV files")
    for member in [*labels, *tables]:
        link = folder / member.name
        if link.is_symlink() and link.resolve() == member.resolve():
            continue
        link.symlink_to(member.resolve())
    return folder


def build_benchmark(sources, work):
    work = broll_ingestion.resolve_work_root(work)
    freeze_dir = work / "benchmark_inputs"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = freeze_dir / "data_freeze.yaml"
    if freeze_path.exists():
        raise FileExistsError(f"Benchmark data has already been built at {freeze_path}")
    for dataset in RAW_FOLDERS:
        source = sources[dataset]
        link_source(work, dataset, organize_cgmacros(work, source) if dataset == cgmacros.DATASET else source)
    tarball = work / "acquisition/Broll_2021/source/iglu_4.2.2.tar.gz"
    tarball.parent.mkdir(parents=True, exist_ok=True)
    if not tarball.is_symlink() or tarball.resolve() != sources["Broll_2021"]:
        tarball.symlink_to(sources["Broll_2021"])
    audit = subprocess.run([
        sys.executable, str(REPO / "scripts/acquisition/audit_broll.py"),
        "--work-root", str(work), "--source-tarball", str(tarball),
    ], check=True, stdout=subprocess.PIPE, text=True)
    audit_path = Path(audit.stdout.strip().splitlines()[-1])
    broll_source = broll_ingestion.run(work, audit_run_id=audit_path.name)
    t1d_runs = {dataset: t1d.run_dataset(work, dataset) for dataset in t1d.DATASETS}
    runs = {
        **t1d_runs,
        ai_readi.DATASET: ai_readi.run(work),
        hall.DATASET: hall.run(work),
        cgmacros.DATASET: cgmacros.run(work),
        broll.DATASET: broll.run(work, broll_source.name),
        "T1D_LODO": t1d_folds.run_folds(work, t1d_runs),
    }
    (freeze_dir / "t1d_folds.yaml").write_bytes((REPO / "configs/t1d_folds.yaml").read_bytes())
    with freeze_path.open("x") as stream:
        yaml.safe_dump({
            "specification_version": "1.5",
            "runs": {dataset: value.name for dataset, value in runs.items()},
        }, stream)
    return freeze_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="JSON paths to the seven original releases")
    parser.add_argument("--work-root", type=Path, required=True, help="External directory for generated data")
    args = parser.parse_args()
    print(build_benchmark(source_paths(args.sources), args.work_root))


if __name__ == "__main__":
    main()
