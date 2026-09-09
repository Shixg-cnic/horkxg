#!/usr/bin/env python3
"""Collect paired Jet and imported-hierarchy runs into a reviewable CSV."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "build-gh200/experiments/results/gpu_lp_jet_compare"


def first(pattern: str, text: str, default: str = "") -> str:
    matches = re.findall(pattern, text)
    return matches[-1] if matches else default


def number(pattern: str, text: str, default: float = float("nan")) -> float:
    value = first(pattern, text)
    return float(value) if value else default


def main() -> None:
    rows = []
    for dataset in ("products", "com-LiveJournal"):
        for k in (4, 8, 32):
            run = RESULTS / dataset / f"k{k}" / "seed0"
            a_log = (run / "A_jet.log").read_text()
            b_log = (run / "B_gpu_lp.log").read_text()
            b_coarsen = (run / "B_gpu_lp_coarsen.log").read_text()
            a = json.loads((run / "A_independent.json").read_text())
            b = json.loads((run / "B_independent.json").read_text())
            a_cut = a["cut"]
            b_cut = b["cut"]
            rows.append({
                "dataset": dataset,
                "k": k,
                "seed": 0,
                "A_cut": a_cut,
                "B_cut": b_cut,
                "B_vs_A_percent": 100.0 * (a_cut - b_cut) / a_cut,
                "A_cut_ratio": a["cut_ratio"],
                "B_cut_ratio": b["cut_ratio"],
                "A_max_load_ratio": a["max_load_ratio"],
                "B_max_load_ratio": b["max_load_ratio"],
                "A_partition_seconds": number(r"Total Partitioning Time: ([0-9.e+-]+)", a_log),
                "A_coarsen_seconds": number(r"Coarsening time: ([0-9.e+-]+)", a_log),
                "A_init_seconds": number(r"Initial partitioning time: ([0-9.e+-]+)", a_log),
                "A_refine_seconds": number(r"Uncoarsening time: ([0-9.e+-]+)", a_log),
                "B_coarsen_seconds": number(r"ml_total_seconds=([0-9.e+-]+)", b_coarsen),
                "B_jet_backend_seconds": number(r"jet_external_total_seconds=([0-9.e+-]+)", b_log),
                "B_coarsest_vertices": first(r"coarsest_vertices=([0-9]+)", b_coarsen),
                "B_levels": first(r"ml_hierarchy_levels=([0-9]+)", b_coarsen),
                "B_stop_reason": first(r"stop_reason=([^ ]+)", b_coarsen),
                "B_capacity_ok": b["capacity_ok"],
            })
    output = RESULTS / "seed0_summary.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    for row in rows:
        print(row["dataset"], row["k"],
              "A", row["A_cut"], "B", row["B_cut"],
              "B_vs_A_percent", f'{row["B_vs_A_percent"]:.3f}',
              "B_levels", row["B_levels"],
              "B_coarsest", row["B_coarsest_vertices"])


if __name__ == "__main__":
    main()
