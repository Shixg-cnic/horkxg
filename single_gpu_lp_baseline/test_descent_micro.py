"""Small-graph correctness test for re-proposed descent micro-batches."""

from array import array
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
BINARY = Path(os.environ.get(
    "MICRO_TEST_BINARY", ROOT / "build-gh200/single_gpu_lp_baseline"))


def make_csr(n, edges):
    rows = [[] for _ in range(n)]
    for u, v in edges:
        rows[u].append(v)
        rows[v].append(u)
    offsets = [0]
    indices = []
    for row in rows:
        indices.extend(sorted(row))
        offsets.append(len(indices))
    return offsets, indices


def main():
    n = 20
    edges = [(i, (i + 1) % n) for i in range(n)]
    edges += [(i, (i + 5) % n) for i in range(0, n, 2)]
    offsets, indices = make_csr(n, edges)
    labels = [0] * 10 + [1] * 10
    with tempfile.TemporaryDirectory(prefix="lp-micro-test-") as directory:
        root = Path(directory)
        for name, code, values in (
            ("ptr", "q", offsets),
            ("idx", "q", indices),
            ("initial", "i", labels),
        ):
            with (root / name).open("wb") as stream:
                array(code, values).tofile(stream)
        environment = os.environ.copy()
        environment.update(
            INITIAL_PARTITION=str(root / "initial"),
            ENABLE_FIELD="0",
            PAIR_EXCHANGE="0",
            BLOCK_LP="0",
            GLOBAL_CYCLES="1",
            BALANCE_ROUNDS="0",
            POLISH_ROUNDS="0",
            DESCENT_MICRO_BATCH="2",
            DESCENT_MICRO_ROUNDS="4",
            INCREMENTAL_CUT="1",
            INCREMENTAL_CUT_VERIFY="1",
        )
        result = subprocess.run(
            [str(BINARY), str(root / "ptr"), str(root / "idx"), "2",
             str(root / "out"), "30", "4", "1", "1.10"],
            env=environment, capture_output=True, text=True, check=True,
        )
        output = array("i")
        with (root / "out").open("rb") as stream:
            output.fromfile(stream, n)
        assert all(label in (0, 1) for label in output)
        loads = [output.count(0), output.count(1)]
        assert max(loads) <= 11
        directed_cut = sum(
            label != output[neighbor]
            for vertex, label in enumerate(output)
            for neighbor in indices[offsets[vertex]:offsets[vertex + 1]]
        )
        reported = int(re.search(r"final_cut=(\d+) ", result.stdout).group(1))
        assert reported == directed_cut
        assert "micro-batch incremental cut verification failed" not in result.stdout
    print("PASS: descent micro-batches preserve labels, capacity and exact CSR cut")


if __name__ == "__main__":
    main()
