#!/usr/bin/env python3
"""Independent CSR cut/load oracle for text or binary int32 partitions."""

from __future__ import annotations

import argparse
from array import array
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path


def read_i64(path: Path) -> array:
    raw = path.read_bytes()
    if len(raw) % 8:
        raise ValueError(f"invalid int64 CSR size: {path}")
    values = array("q")
    values.frombytes(raw)
    return values


def verify(indptr: Path, indices_path: Path, partition: Path, parts: int,
           imbalance: str = "1.03", partition_format: str = "binary") -> dict:
    if not 2 <= parts <= 32 or not Decimal(imbalance).is_finite() or Decimal(imbalance) < 1:
        raise ValueError("invalid parts or imbalance ratio")
    offsets, indices = read_i64(indptr), read_i64(indices_path)
    if not offsets or offsets[0] != 0 or offsets[-1] != len(indices):
        raise ValueError("invalid CSR endpoints")
    if any(a > b or a < 0 for a, b in zip(offsets, offsets[1:])):
        raise ValueError("nonmonotone CSR offsets")
    n = len(offsets) - 1
    raw = partition.read_bytes()
    if partition_format == "text":
        labels = [int(x) for x in raw.decode().split()]
    else:
        if len(raw) != n * 4:
            raise ValueError("partition byte length does not match vertices")
        labels = array("i")
        labels.frombytes(raw)
    if len(labels) != n:
        raise ValueError("partition label count does not match vertices")
    loads = [0] * parts
    for label in labels:
        if not 0 <= label < parts:
            raise ValueError(f"invalid label: {label}")
        loads[label] += 1
    directed_cut = 0
    nonloop_entries = 0
    for v in range(n):
        for edge in range(offsets[v], offsets[v + 1]):
            u = indices[edge]
            if not 0 <= u < n:
                raise ValueError("invalid neighbor id")
            nonloop_entries += u != v
            directed_cut += labels[v] != labels[u]
    if directed_cut % 2 or nonloop_entries % 2:
        raise ValueError("odd undirected edge/cut count; input must be symmetric")
    capacity = int((Decimal(imbalance) * n / parts).to_integral_value(rounding=ROUND_FLOOR))
    cut, edges = directed_cut // 2, nonloop_entries // 2
    return {
        "vertices": n, "undirected_edges": edges, "cut": cut,
        "cut_ratio": cut / edges if edges else 0.0, "loads": loads,
        "max_load": max(loads), "max_load_ratio": max(loads) * parts / n if n else 0.0,
        "capacity": capacity, "capacity_ok": max(loads) <= capacity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indptr", required=True, type=Path)
    parser.add_argument("--indices", required=True, type=Path)
    parser.add_argument("--partition", required=True, type=Path)
    parser.add_argument("--parts", required=True, type=int)
    parser.add_argument("--imbalance-ratio", default="1.03")
    parser.add_argument("--partition-format", choices=("binary", "text"), default="binary")
    args = parser.parse_args()
    result = verify(args.indptr, args.indices, args.partition, args.parts,
                    args.imbalance_ratio, args.partition_format)
    print(json.dumps(result, sort_keys=True))
    if not result["capacity_ok"]:
        raise SystemExit("partition violates the requested capacity")


if __name__ == "__main__":
    main()
