#!/usr/bin/env python3
"""Independent original-CSR cut and capacity verifier."""

from __future__ import annotations

import argparse
import json
import mmap
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indptr", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--parts", type=int, required=True)
    args = parser.parse_args()

    labels = [int(x) for x in Path(args.partition).read_text().split()]
    with open(args.indptr, "rb") as offset_file, open(args.indices, "rb") as index_file:
        with mmap.mmap(offset_file.fileno(), 0, access=mmap.ACCESS_READ) as offset_map:
            with mmap.mmap(index_file.fileno(), 0, access=mmap.ACCESS_READ) as index_map:
                offsets = memoryview(offset_map).cast("q")
                indices = memoryview(index_map).cast("q")
                n = len(offsets) - 1
                if len(labels) != n:
                    raise SystemExit(f"partition length {len(labels)} != vertices {n}")
                loads = [0] * args.parts
                for label in labels:
                    if label < 0 or label >= args.parts:
                        raise SystemExit(f"invalid label {label}")
                    loads[label] += 1
                directed_cut = 0
                for v in range(n):
                    label = labels[v]
                    for edge in range(offsets[v], offsets[v + 1]):
                        if label != labels[indices[edge]]:
                            directed_cut += 1
                if directed_cut % 2:
                    raise SystemExit("directed cut is odd; CSR is not symmetric")
                cut = directed_cut // 2
                undirected_edges = len(indices) // 2
                # Jet's documented/statistical convention is ceil(n/k) as the
                # optimal integer part size, then floor(1.10*optimal_size).
                # Report the raw ratio as well, but use that same integer
                # capacity for the pass/fail check.
                optimal_size = (n + args.parts - 1) // args.parts
                capacity = int(1.10 * optimal_size)
                max_ratio = max(loads) / (n / args.parts)
                print(json.dumps({
                    "vertices": n,
                    "undirected_edges": undirected_edges,
                    "cut": cut,
                    "cut_ratio": cut / undirected_edges if undirected_edges else 0.0,
                    "max_load": max(loads),
                    "max_load_ratio": max_ratio,
                    "capacity": capacity,
                    "capacity_ok": max(loads) <= capacity,
                }, sort_keys=True))
                indices.release()
                offsets.release()


if __name__ == "__main__":
    main()
