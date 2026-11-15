"""Use a locally extracted official Nutrition5k RGB release to prepare model inputs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.n5k.metadata import read_ids


def extract_keyframes(video, output):
    output.mkdir(parents=True, exist_ok=True)
    short_side = 320
    scale = f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)':flags=area"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-threads", "1",
        "-skip_frame", "nokey", "-i", str(video), "-vsync", "vfr", "-vf", scale,
        "-q:v", "2", "-start_number", "0", "-y", str(output / f"{video.stem}_frame_%d.jpg"),
    ], check=True)


def prepare_dish(dish, raw, output):
    overhead = raw / "imagery" / "realsense_overhead" / dish / "rgb.png"
    if overhead.is_file():
        link = output / "overhead" / dish / "rgb.png"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(overhead.resolve())
    for video in sorted((raw / "imagery" / "side_angles" / dish).glob("camera_*.h264")):
        extract_keyframes(video, output / "side_frames" / dish)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    raw, output = args.raw.resolve(), args.out.resolve()
    split_root = raw / "dish_ids/splits"
    dishes = sorted(set(read_ids(split_root / "rgb_train_ids.txt")) |
                    set(read_ids(split_root / "rgb_test_ids.txt")))
    if output.is_relative_to(raw):
        parser.error("Write prepared images outside the original dataset directory.")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("dish_ids", "metadata"):
        (output / directory).symlink_to(raw / directory, target_is_directory=True)
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        for _ in workers.map(prepare_dish, dishes, repeat(raw), repeat(output)):
            pass


if __name__ == "__main__":
    main()
