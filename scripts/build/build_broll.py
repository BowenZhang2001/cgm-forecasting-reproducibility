#!/usr/bin/env python3
"""Build the whole-dataset external Broll CGM and common-origin manifests."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cgm_pipeline.build.broll import main  # noqa: E402


if __name__ == "__main__":
    main()
