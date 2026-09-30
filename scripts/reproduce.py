#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--study",
    choices=["study1", "study2", "all"],
    default="all",
)
arguments = parser.parse_args()

subprocess.run(
    [sys.executable, str(ROOT / "scripts/unpack_data.py")],
    check=True,
)

if arguments.study in {"study1", "all"}:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "study1/analysis/run_analysis.py"),
            "--output",
            str(ROOT / "reproduced/study1"),
        ],
        cwd=ROOT,
        check=True,
    )

if arguments.study in {"study2", "all"}:
    environment = dict(
        os.environ,
        REVISION_OUTPUT_DIR=str(ROOT / "reproduced/study2"),
    )
    subprocess.run(
        [sys.executable, str(ROOT / "study2/code/analyze.py")],
        cwd=ROOT,
        env=environment,
        check=True,
    )
