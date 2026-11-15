"""Train the ConvNeXt-Base checkpoints consumed by CGMacros forecasting."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


SCRIPTS = Path(__file__).resolve().parent
SETTINGS = SCRIPTS.parents[1] / "configs/image_encoders.json"


def run(name, arguments, config):
    command = [sys.executable, str(SCRIPTS / f"train_{name}.py"), *arguments]
    for key, value in config.items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--food101", type=Path, required=True)
    parser.add_argument("--nutrition5k", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    configurations = json.loads(SETTINGS.read_text())
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "nutrition5k-inputs"
    subprocess.run([
        sys.executable, str(SCRIPTS / "prepare_nutrition5k.py"),
        "--raw", str(args.nutrition5k.resolve()), "--out", str(prepared),
    ], check=True)
    food = output / "food101"
    run("food101", ["--root", str(args.food101.resolve()), "--out", str(food)],
        configurations["food101"])
    run("nutrition5k", ["--n5k-root", str(prepared), "--init", str(food / "best.pt"),
                       "--out", str(output / "nutrition5k")], configurations["nutrition5k"])


if __name__ == "__main__":
    main()
