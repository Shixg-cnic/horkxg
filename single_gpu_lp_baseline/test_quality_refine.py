"""Check exact output cuts, capacity, and non-regression on random symmetric graphs."""
from array import array
from pathlib import Path
import random
import re
import subprocess
import tempfile

binary = Path(__file__).resolve().parent / "build-gh200/quality_refine"
with tempfile.TemporaryDirectory(prefix="lp-quality-test-") as directory:
    root = Path(directory)
    for seed in range(20):
        rng = random.Random(seed)
        n, k = 48, 4
        neighbors = [[] for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < (0.5 if u // 12 == v // 12 else 0.04):
                    neighbors[u].append(v)
                    neighbors[v].append(u)
        labels = [v % k for v in range(n)]
        rng.shuffle(labels)
        offsets, indices = [0], []
        for row in neighbors:
            indices.extend(row)
            offsets.append(len(indices))
        for name, code, values in [("ptr", "q", offsets), ("idx", "q", indices), ("in", "i", labels)]:
            with (root / name).open("wb") as f:
                array(code, values).tofile(f)
        for group in [0, 8, 32]:
            ratio = 1.0 if group == 0 or seed % 2 else 1.25
            run = subprocess.run([str(binary), str(root / "ptr"), str(root / "idx"), str(root / "in"), str(root / "out"), str(k), str(ratio), "5", "100", str(seed), "1", str(group)], capture_output=True, text=True, check=True)
            result = array("i")
            with (root / "out").open("rb") as f:
                result.fromfile(f, n)
            def cut(part):
                return sum(part[u] != part[v] for u in range(n) for v in neighbors[u]) // 2
            assert all(result.count(p) <= int(n / k * ratio) for p in range(k))
            assert cut(result) <= cut(labels)
            assert cut(result) == int(re.search(r"final_cut=(\d+)", run.stdout).group(1))
    print("PASS: 60 balanced random-graph runs; exact cuts and non-regression")
