#!/usr/bin/env python3
"""Generate symmetric CSR fixtures and independently check full GPU partitioning."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_partition import verify


def generate(root: Path, kind: str, n: int, seed: int = 17) -> tuple[Path, Path]:
    rng = random.Random(seed)
    rows: list[set[int]] = [set() for _ in range(n)]

    def add(u: int, v: int) -> None:
        if u != v:
            rows[u].add(v)
            rows[v].add(u)

    communities = 8
    size = n // communities
    for v in range(n):
        if kind == "social":
            base = v // size * size
            # Triangles, communities, hubs, and weak cross-community edges.
            add(v, base)
            add(v, base + (v - base + 1) % size)
            for _ in range(8):
                add(v, base + rng.randrange(size))
            if rng.random() < 0.6:
                add(v, rng.randrange(n))
        elif kind == "circuit":
            # Local connectivity and sparse long-range nets; no claim that
            # this approximates an actual circuit timing benchmark.
            for distance in (1, 2, 4):
                if v + distance < n:
                    add(v, v + distance)
            if v % 16 == 0:
                add(v, rng.randrange(n))
        elif kind == "disconnected":
            if v % size:
                add(v - 1, v)
        else:
            raise ValueError(kind)
    root.mkdir(parents=True, exist_ok=True)
    offsets, indices = array("q", [0]), array("q")
    for row in rows:
        indices.extend(sorted(row))
        offsets.append(len(indices))
    indptr, index_path = root / "indptr.bin", root / "indices.bin"
    indptr.write_bytes(offsets.tobytes())
    index_path.write_bytes(indices.tobytes())
    return indptr, index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build-5090"))
    parser.add_argument("--output", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--vertices", type=int, default=16384)
    parser.add_argument("--parts", type=int, nargs="+", default=[2, 4, 8, 32])
    parser.add_argument("--kinds", nargs="+", default=["social", "circuit", "disconnected"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=["standard", "big"])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.vertices < 32 or args.vertices % 8 or args.repeats < 1:
        parser.error("vertices must be a multiple of eight >=32; repeats must be positive")
    results = []
    for kind in args.kinds:
        root = args.output / kind
        indptr, indices = generate(root, kind, args.vertices)
        for k in args.parts:
            hashes = {}
            for variant in args.variants:
                for repeat in range(args.repeats):
                    name = f"k{k}-{variant}-{repeat}"
                    partition, log = root / f"{name}.part", root / f"{name}.log"
                    command = [str((args.build / f"gpart_partition_{variant}").resolve()),
                               str(indptr), str(indices), str(k), str(partition), "1.03", "0"]
                    if args.verify:
                        command.append("--verify")
                    with log.open("w") as stream:
                        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                    result = verify(indptr, indices, partition, k)
                    if not result["capacity_ok"]:
                        raise AssertionError(result)
                    digest = hashlib.sha256(partition.read_bytes()).hexdigest()
                    same_labels=not hashes or digest == next(iter(hashes.values()))
                    hashes[name] = digest
                    text = log.read_text()
                    reported = int(re.findall(r"^final_edge_cut=(\d+)$", text, re.M)[-1])
                    if reported != result["cut"]:
                        raise AssertionError(f"reported cut differs from independent oracle: {name}")
                    timings = re.findall(r"partition_timing (.*)", text)[-1]
                    result.update(kind=kind, parts=k, variant=variant, repeat=repeat,
                                  sha256=digest, verify=args.verify,
                                  same_labels_as_first=same_labels,
                                  timing={key: float(value) for key, value in
                                          re.findall(r"(\w+_seconds)=([0-9.e+-]+)", timings)})
                    results.append(result)
                    print(json.dumps(result, sort_keys=True), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
