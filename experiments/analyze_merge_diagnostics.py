#!/usr/bin/env python3
"""Summarize accepted SCLP merges and compare MR with priority orientation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path
from typing import Iterator


RECORD = struct.Struct("<6I10Q")
FIELDS = (
    "level", "round", "vertex", "source", "target", "reserved",
    "acur", "a1", "a2", "weighted_degree", "edge_degree",
    "vertex_weight", "target_weight", "capacity", "jet_loss", "metis_loss",
)
NORMALIZED_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30,
                    0.40, 0.50, 0.60, 0.80, 1.0000000001]


def records(path: Path) -> Iterator[dict[str, int]]:
    size = path.stat().st_size
    if size % RECORD.size:
        raise ValueError(f"{path}: size is not a multiple of {RECORD.size}")
    with path.open("rb") as stream:
        while block := stream.read(RECORD.size * 65536):
            if len(block) % RECORD.size:
                raise ValueError(f"{path}: truncated record block")
            for values in struct.iter_unpack(RECORD.format, block):
                yield dict(zip(FIELDS, values))


def feature_values(record: dict[str, int]) -> dict[str, float]:
    degree = record["weighted_degree"]
    denominator = float(degree) if degree else 1.0
    return {
        "normalized_gain": (record["a1"] - record["acur"]) / denominator,
        "best_second_margin": (record["a1"] - record["a2"]) / denominator,
        "support_ratio": record["a1"] / denominator,
        "target_capacity_ratio": record["target_weight"] / record["capacity"],
        "edge_degree": float(record["edge_degree"]),
    }


def bucket(feature: str, value: float) -> int:
    if feature == "edge_degree":
        return min(9, int(math.log2(max(1.0, value))))
    for index in range(10):
        if NORMALIZED_EDGES[index] <= value < NORMALIZED_EDGES[index + 1]:
            return index
    return 9


def empty_stats() -> dict:
    return {
        "accepted": 0,
        "oracle": {
            "jet": {"bad": 0, "loss": 0},
            "metis": {"bad": 0, "loss": 0},
        },
        "feature_sums": {
            name: 0.0 for name in (
                "normalized_gain", "best_second_margin", "support_ratio",
                "target_capacity_ratio", "edge_degree"
            )
        },
        "bins": {
            name: [
                {"accepted": 0, "jet_bad": 0, "metis_bad": 0,
                 "jet_loss": 0, "metis_loss": 0}
                for _ in range(10)
            ]
            for name in (
                "normalized_gain", "best_second_margin", "support_ratio",
                "target_capacity_ratio", "edge_degree"
            )
        },
    }


def add(stats: dict, record: dict[str, int]) -> None:
    stats["accepted"] += 1
    for oracle in ("jet", "metis"):
        loss = record[f"{oracle}_loss"]
        stats["oracle"][oracle]["bad"] += loss > 0
        stats["oracle"][oracle]["loss"] += loss
    for feature, value in feature_values(record).items():
        stats["feature_sums"][feature] += value
        item = stats["bins"][feature][bucket(feature, value)]
        item["accepted"] += 1
        for oracle in ("jet", "metis"):
            loss = record[f"{oracle}_loss"]
            item[f"{oracle}_bad"] += loss > 0
            item[f"{oracle}_loss"] += loss


def finish(stats: dict) -> dict:
    accepted = stats["accepted"]
    stats["feature_means"] = {
        name: total / accepted if accepted else 0.0
        for name, total in stats.pop("feature_sums").items()
    }
    for oracle in ("jet", "metis"):
        item = stats["oracle"][oracle]
        item["bad_ratio"] = item["bad"] / accepted if accepted else 0.0
        item["average_loss"] = item["loss"] / accepted if accepted else 0.0
    for feature, bins in stats["bins"].items():
        for index, item in enumerate(bins):
            count = item["accepted"]
            item["range"] = (
                f"2^{index}..2^{index + 1}" if feature == "edge_degree"
                else [NORMALIZED_EDGES[index], NORMALIZED_EDGES[index + 1]]
            )
            for oracle in ("jet", "metis"):
                item[f"{oracle}_bad_ratio"] = (
                    item[f"{oracle}_bad"] / count if count else 0.0
                )
    return stats


def first_round_keys(path: Path) -> set[int]:
    return {
        (record["vertex"] << 32) | record["target"]
        for record in records(path)
        if record["level"] == 0 and record["round"] == 0
    }


def load_levels(prefix: Path) -> list[dict[str, str]]:
    with Path(f"{prefix}.levels.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr-prefix", type=Path, required=True)
    parser.add_argument("--priority-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mr_path = Path(f"{args.mr_prefix}.merges.bin")
    priority_path = Path(f"{args.priority_prefix}.merges.bin")
    mr_first = first_round_keys(mr_path)
    priority_first = first_round_keys(priority_path)
    categories = {
        "mr_all": empty_stats(),
        "priority_all": empty_stats(),
        "first_round_common": empty_stats(),
        "first_round_priority_extra": empty_stats(),
        "first_round_mr_only": empty_stats(),
    }
    for record in records(mr_path):
        add(categories["mr_all"], record)
        if record["level"] == 0 and record["round"] == 0:
            key = (record["vertex"] << 32) | record["target"]
            add(categories[
                "first_round_common" if key in priority_first
                else "first_round_mr_only"
            ], record)
    for record in records(priority_path):
        add(categories["priority_all"], record)
        if record["level"] == 0 and record["round"] == 0:
            key = (record["vertex"] << 32) | record["target"]
            if key not in mr_first:
                add(categories["first_round_priority_extra"], record)

    result = {
        "record_size": RECORD.size,
        "comparison_scope": (
            "common/extra sets use level 0 round 0 only; later coarse states diverge"
        ),
        "set_counts": {
            "mr_first_round": len(mr_first),
            "priority_first_round": len(priority_first),
            "common": len(mr_first & priority_first),
            "priority_extra": len(priority_first - mr_first),
            "mr_only": len(mr_first - priority_first),
        },
        "categories": {name: finish(stats) for name, stats in categories.items()},
        "levels": {
            "mover_receiver": load_levels(args.mr_prefix),
            "priority": load_levels(args.priority_prefix),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "set_counts": result["set_counts"],
        "priority_extra": result["categories"]["first_round_priority_extra"],
    }, indent=2))


if __name__ == "__main__":
    main()
