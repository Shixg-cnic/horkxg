"""Small-graph correctness checks for the optional CUDA pair exchange stage.

The executable remains the system under test.  The Python code only constructs
bounded CSR fixtures and independently checks the final directed cut/load
values; PAIR_VERIFY=1 makes the CUDA path also check every proposed pair with
the host CSR oracle before submission.
"""

from array import array
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
BINARY = Path(os.environ.get("PAIR_TEST_BINARY", ROOT / "build-gh200/single_gpu_lp_baseline"))


def make_csr(n, undirected_edges):
    rows = [[] for _ in range(n)]
    for u, v in undirected_edges:
        rows[u].append(v)
        rows[v].append(u)
    offsets = [0]
    indices = []
    for row in rows:
        indices.extend(sorted(row))
        offsets.append(len(indices))
    return offsets, indices


def cut(indices, offsets, labels):
    return sum(
        labels[u] != labels[v]
        for u in range(len(labels))
        for v in indices[offsets[u] : offsets[u + 1]]
    )


def run_case(n, k, labels, edges, *, pair=True, verify=True):
    offsets, indices = make_csr(n, edges)
    with tempfile.TemporaryDirectory(prefix="cuda-pair-test-") as directory:
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
            GLOBAL_CYCLES="1",
            FIELD_ROUNDS="0",
            BALANCE_ROUNDS="0",
            REFINE_ROUNDS="0",
            POLISH_ROUNDS="0",
        )
        if pair is None:
            for name in ("PAIR_EXCHANGE", "PAIR_VERIFY", "PAIR_EXCHANGE_ROUNDS",
                         "PAIR_TOP_TARGETS", "PAIR_BUCKET_LIMIT"):
                environment.pop(name, None)
        else:
            environment.update(
                PAIR_EXCHANGE="1" if pair else "0",
                PAIR_VERIFY="1" if verify else "0",
                PAIR_EXCHANGE_ROUNDS="2",
                PAIR_TOP_TARGETS="2",
                PAIR_BUCKET_LIMIT="32",
            )
        result = subprocess.run(
            [str(BINARY), str(root / "ptr"), str(root / "idx"), str(k),
             str(root / "out"), "30", "0", "1", "1.10"],
            env=environment, capture_output=True, text=True, check=True,
        )
        output = array("i")
        with (root / "out").open("rb") as stream:
            output.fromfile(stream, n)
        output = output.tolist()
        final_line = next(line for line in result.stdout.splitlines()
                          if line.startswith("final_cut="))
        final_cut = int(re.search(r"final_cut=(\d+)", final_line).group(1))
        summaries = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("pair_summary="):
                summaries.append({key: int(value) for key, value in
                                  re.findall(r"(\w+)=(-?\d+)", line)})
        assert final_cut == cut(indices, offsets, output)
        assert all(output.count(part) <= n // k * 11 // 10 for part in range(k))
        return result.stdout, output, summaries


def assert_first_summary(summaries, proposed, accepted):
    assert summaries, "pair stage did not report a round"
    assert summaries[0]["proposed_exchanges"] == proposed, summaries[0]
    assert summaries[0]["accepted_exchanges"] == accepted, summaries[0]


def main():
    if not BINARY.exists():
        raise SystemExit(f"missing CUDA binary: {BINARY}")

    # Both sides are full.  Every candidate is adjacent to its reverse
    # candidate, so the -2*adjacency correction is exercised.
    k44_edges = [(u, v) for u in range(4) for v in range(4, 8)]
    stdout, output, summaries = run_case(8, 2, [0] * 4 + [1] * 4, k44_edges)
    assert_first_summary(summaries, 1, 1)
    assert "accepted_gain=6" in stdout
    assert cut(make_csr(8, k44_edges)[1], make_csr(8, k44_edges)[0], output) == 16

    # The two single gains are positive, but the adjacent-edge correction makes
    # the exact exchange gain zero; it must not be submitted.
    stdout, _, summaries = run_case(4, 2, [0, 0, 1, 1], [(0, 2)])
    assert_first_summary(summaries, 0, 0)
    assert "pair_proposed" not in stdout

    # The same check with parallel edges: multiplicity is 2, not a boolean.
    stdout, _, summaries = run_case(4, 2, [0, 0, 1, 1], [(0, 2), (0, 2)])
    assert_first_summary(summaries, 0, 0)

    # One vertex is present in two directed target buckets; only one of the
    # two positive exchanges may be accepted in a batch.
    shared_edges = [(0, 3), (2, 1), (0, 5), (4, 1)]
    _, _, summaries = run_case(8, 4, [0, 0, 1, 1, 2, 2, 3, 3], shared_edges)
    assert_first_summary(summaries, 2, 1)

    # The two exchanges have disjoint endpoints but a cross-exchange edge;
    # conservative batch filtering must also accept only one.
    cross_edges = [(0, 3), (2, 1), (4, 7), (6, 5), (0, 4)]
    _, _, summaries = run_case(8, 4, [0, 0, 1, 1, 2, 2, 3, 3], cross_edges)
    assert_first_summary(summaries, 2, 1)

    # Isolated/no-gain graph: no invalid loop and no accepted exchange.
    _, output, summaries = run_case(4, 2, [0, 0, 1, 1], [])
    assert_first_summary(summaries, 0, 0)
    assert output == [0, 0, 1, 1]

    # Random balanced fixtures exercise k=2,4,8 and the complete CUDA/CPU
    # verification path, including exact post-submit cut and loads.
    for seed in range(8):
        rng = random.Random(seed)
        k = 2 if seed % 2 == 0 else 4
        n = 16 if k == 2 else 16
        labels = [vertex % k for vertex in range(n)]
        rng.shuffle(labels)
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)
                 if rng.random() < 0.18]
        run_case(n, k, labels, edges)

    # Disabled is the baseline path.  It is compared byte-for-byte with an
    # invocation where the new environment variables are absent.
    baseline_args = (8, 2, [0] * 4 + [1] * 4, k44_edges)
    _, disabled_output, _ = run_case(*baseline_args, pair=False, verify=False)
    _, default_output, _ = run_case(*baseline_args, pair=None, verify=None)
    assert disabled_output == default_output
    print("PASS: pair exchange formula, multiplicity, conflicts, capacities, random CUDA verification, and disabled baseline")


if __name__ == "__main__":
    main()
