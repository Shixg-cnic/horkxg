#!/usr/bin/env python3
"""Small independent oracle for the GPU coarsening hierarchy exporter.

The test creates a symmetric CSR with an isolated component, parallel edges,
and self-loops, then checks every emitted level without importing the C++
implementation's internal state.
"""

from __future__ import annotations

import os
import random
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def write_i64(path: Path, values: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}q", *values))


def make_graph() -> tuple[list[int], list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(256)]
    def add(u: int, v: int, copies: int = 1) -> None:
        for _ in range(copies):
            adjacency[u].append(v)
            adjacency[v].append(u)

    # Parallel edges and self-loops in one component.
    adjacency[0].append(0)
    add(0, 1, 2)
    add(1, 2)
    add(2, 3)
    # A disconnected component.
    add(4, 5)
    add(5, 6, 2)
    # An isolated vertex (7) and a separate pair (8, 9).
    add(8, 9)
    # The remainder is a collection of short chains, ensuring n > cutoff.
    for u in range(10, 255, 2):
        add(u, u + 1)
    offsets = [0]
    indices: list[int] = []
    for row in adjacency:
        row.sort()
        indices.extend(row)
        offsets.append(len(indices))
    return offsets, indices


def read_i32(f, count: int) -> list[int]:
    raw = f.read(4 * count)
    if len(raw) != 4 * count:
        raise AssertionError("truncated hierarchy")
    return list(struct.unpack(f"<{count}i", raw))


def cut(rows: list[int], cols: list[int], weights: list[int], labels: list[int]) -> int:
    directed = 0
    for v in range(len(rows) - 1):
        for e in range(rows[v], rows[v + 1]):
            if labels[v] != labels[cols[e]]:
                directed += weights[e]
    if directed % 2:
        raise AssertionError("directed cut is not even")
    return directed // 2


def check(path: Path, capacities: list[int]) -> None:
    # Sequential parser matching the exporter format: each level's map is
    # stored immediately after that level's vertex weights, and maps level l-1
    # vertices to level l.  Keep the raw levels first for clarity.
    with path.open("rb") as f:
        count = read_i32(f, 1)[0]
        levels = []
        for level in range(count):
            n, m = read_i32(f, 2)
            rows = read_i32(f, n + 1)
            cols = read_i32(f, m)
            ew = read_i32(f, m)
            vw = read_i32(f, n)
            mapping = read_i32(f, levels[-1][0]) if level else None
            levels.append((n, m, rows, cols, ew, vw, mapping))
        assert f.read() == b""
    assert len(levels) >= 2
    rng = random.Random(17)
    for level, (n, m, rows, cols, ew, vw, mapping) in enumerate(levels):
        assert rows[0] == 0 and rows[-1] == m
        assert all(rows[i] <= rows[i + 1] for i in range(n))
        assert all(0 <= u < n for u in cols)
        assert all(w > 0 for w in ew) and all(w > 0 for w in vw)
        pairs = {(v, cols[e]): ew[e] for v in range(n) for e in range(rows[v], rows[v + 1])}
        for (u, v), weight in pairs.items():
            if level > 0:
                assert u != v
            assert pairs.get((v, u)) == weight
        if mapping is not None:
            fn, _, frows, fcols, few, fvw, _ = levels[level - 1]
            assert len(mapping) == fn and all(0 <= c < n for c in mapping)
            assert len(set(mapping)) == n
            sums = [0] * n
            for v, c in enumerate(mapping):
                sums[c] += fvw[v]
            assert sums == vw
            labels = [rng.randrange(7) for _ in range(n)]
            projected = [labels[c] for c in mapping]
            assert cut(frows, fcols, few, projected) == cut(rows, cols, ew, labels)
            assert max(vw) <= capacities[level - 1]


def main() -> None:
    binary = Path(os.environ.get("MULTILEVEL_BIN", "build-gh200/multilevel_lp"))
    method = os.environ.get("BASC_TEST_METHOD", "basc")
    if method not in {"basc", "basc_gpu", "frontier", "sclp"}:
        raise SystemExit("BASC_TEST_METHOD must be basc, basc_gpu, frontier, or sclp")
    with tempfile.TemporaryDirectory(prefix="basc-small-") as tmp:
        root = Path(tmp)
        offsets, indices = make_graph()
        write_i64(root / "indptr.bin", offsets)
        write_i64(root / "indices.bin", indices)
        for k in (1, 2, 4):
            out = root / f"basc-k{k}.hierarchy"
            log = root / f"basc-k{k}.log"
            child_env = os.environ.copy()
            # beta=64 intentionally gives capacity one on this 256-vertex
            # fixture.  Use a smaller beta to exercise weighted admission and
            # contraction; production defaults remain unchanged.
            if method == "sclp":
                child_env.setdefault("SCLP_BETA", "1")
            stop_ratio = "1.0" if method == "sclp" else "0.85"
            subprocess.run(
                [str(binary), str(root / "indptr.bin"), str(root / "indices.bin"),
                 "4", str(out), "1.10", "0", stop_ratio, method, str(k)],
                check=True, stdout=log.open("w"), stderr=subprocess.STDOUT,
                env=child_env,
            )
            text = log.read_text()
            capacities = [int(x) for x in re.findall(
                r"ml_gpu_(?:basc(?:_device)?|frontier|sclp) .*?cluster_cap=(\d+)", text)]
            try:
                check(out, capacities)
            except Exception:
                print(text, file=sys.stderr)
                raise
            assert "projection_cut=ok" in text
    print(f"{method} small hierarchy, capacity, symmetry, coverage, and cut tests passed")


if __name__ == "__main__":
    main()
