#!/usr/bin/env python3
"""Repeat one BASC run with the same seed and compare hierarchy bytes/logs.

Persistent artifacts are written under the sibling single_gpu_lp_baseline
results directory by default, never under this project checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT.parent / (
    "single_gpu_lp_baseline/build-gh200/experiments/results/basc_repeatability"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indptr", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--parts", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--basc-k", type=int, default=2)
    parser.add_argument(
        "--method", choices=("basc", "basc_gpu", "frontier"), default="basc"
    )
    parser.add_argument("--stop-ratio", type=float, default=0.85)
    parser.add_argument("--max-levels", type=int, default=24)
    parser.add_argument(
        "--binary", default=str(PROJECT_ROOT / "build-gh200/multilevel_lp")
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2")

    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.method if args.output_dir == str(DEFAULT_RESULTS) else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    command = [
        args.binary,
        args.indptr,
        args.indices,
        str(args.parts),
        "PLACEHOLDER",
        "1.10",
        str(args.seed),
        str(args.stop_ratio),
        args.method,
        str(args.basc_k),
        str(args.max_levels),
    ]
    for repeat in range(args.repeats):
        hierarchy = output_dir / f"repeat{repeat}.hierarchy"
        log_path = output_dir / f"repeat{repeat}.log"
        run_command = list(command)
        run_command[4] = str(hierarchy)
        with log_path.open("w") as log:
            subprocess.run(
                run_command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        log_text = log_path.read_text()
        records.append(
            {
                "repeat": repeat,
                "hierarchy": str(hierarchy),
                "sha256": sha256(hierarchy),
                "basc_levels": len(re.findall(
                    r"^ml_gpu_(?:basc(?:_device)?|frontier) level=",
                    log_text, re.MULTILINE)),
                "coarsen_seconds": float(
                    re.findall(r"ml_total_seconds=([0-9.e+-]+)", log_text)[-1]
                ),
            }
        )

    summary = {
        "command": command[:4] + command[5:],
        "repeats": records,
        "identical_hierarchy": len({record["sha256"] for record in records}) == 1,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["identical_hierarchy"]:
        raise SystemExit(f"{args.method} hierarchy is not repeatable for the same seed")


if __name__ == "__main__":
    main()
