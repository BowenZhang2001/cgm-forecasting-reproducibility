#!/usr/bin/env python3
"""Run verified Broll source ingestion without cleaning or changing the source clock."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cgm_pipeline.ingestion.broll import main  # noqa: E402


if __name__ == "__main__":
    main()
