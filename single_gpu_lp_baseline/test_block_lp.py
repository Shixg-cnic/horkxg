"""Small-graph correctness checks for the optional CUDA block-LP stage.

The Python code constructs bounded CSR fixtures and independently checks the
final labels, directed cut, and capacity.  BLOCK_VERIFY=1 additionally makes
the C++ host verifier check every generated connected candidate and the trial
labels before submission.
"""

from array import array
import math
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
BINARY = Path(os.environ.get(
    "BLOCK_TEST_BINARY", ROOT / "build-gh200/single_gpu_lp_baseline"))


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
        for v in indices[offsets[u]:offsets[u + 1]]
    )


def run_case(n, k, labels, edges, *, block=True, verify=True,
             max_size=16):
    offsets, indices = make_csr(n, edges)
    with tempfile.TemporaryDirectory(prefix="cuda-block-test-") as directory:
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
            GLOBAL_CYCLES="0",
            FIELD_ROUNDS="0",
            BALANCE_ROUNDS="0",
            REFINE_ROUNDS="0",
            POLISH_ROUNDS="0",
            PAIR_EXCHANGE="0",
            BLOCK_MAX_SIZE=str(max_size),
            BLOCK_SEEDS_PER_PAIR="8",
            BLOCK_FRONTIER_LIMIT="256",
            BLOCK_ROUNDS="2",
        )
        if block is None:
            for name in ("BLOCK_LP", "BLOCK_VERIFY", "BLOCK_MAX_SIZE",
                         "BLOCK_SEEDS_PER_PAIR", "BLOCK_FRONTIER_LIMIT",
                         "BLOCK_ROUNDS"):
                environment.pop(name, None)
        elif block:
            environment.update(
                BLOCK_LP="1",
                BLOCK_VERIFY="1" if verify else "0",
            )
        else:
            environment["BLOCK_LP"] = "0"
            environment["BLOCK_VERIFY"] = "0"
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
        block_lines = [line.strip() for line in result.stdout.splitlines()
                       if line.strip().startswith("block_round=")]
        summaries = []
        for line in block_lines:
            values = {}
            for key, value in re.findall(r"(\w+)=([^ ]+)", line):
                try:
                    values[key] = int(value)
                except ValueError:
                    values[key] = float(value)
            summaries.append(values)
        assert final_cut == cut(indices, offsets, output)
        capacity = math.floor(n * 1.10 / k + 1.0e-9)
        assert all(output.count(part) <= capacity for part in range(k))
        return result.stdout, output, summaries


def main():
    if not BINARY.exists():
        raise SystemExit(f"missing CUDA binary: {BINARY}")

    # A and B are full/near-full, but moving the connected pair has gain 2;
    # each single point has gain zero.
    barrier_edges = [(0, 1), (0, 11), (1, 12)]
    stdout, output, summaries = run_case(
        20, 2, [0] * 11 + [1] * 9, barrier_edges)
    assert summaries[0]["size_ge2_positive"] > 0, summaries[0]
    assert summaries[0]["accepted"] == 1, summaries[0]
    assert summaries[0]["batch_gain"] == 2, summaries[0]
    assert output[0:2] == [1, 1], output
    assert "block_polish=" not in stdout

    # Negative seed gains can be crossed: two parallel internal edges make
    # each seed gain -1, while the pair gain is +2.
    negative_seed_edges = [
        (0, 1), (0, 1), (0, 11), (1, 12),
    ]
    _, output, summaries = run_case(
        20, 2, [0] * 11 + [1] * 9, negative_seed_edges, max_size=4)
    assert summaries[0]["nonpositive_seed_positive"] > 0, summaries[0]
    assert summaries[0]["accepted"] == 1, summaries[0]
    assert output[0:2] == [1, 1], output

    # A positive size-two prefix is retained even though growth is allowed to
    # continue.  The summary also reports the size of the highest-gain
    # candidate, which need not be the largest positive candidate on a tie.
    prefix_edges = [
        (0, 1), (0, 11), (1, 12),
        (2, 1), (2, 3), (2, 4),
    ]
    _, _, summaries = run_case(
        20, 2, [0] * 11 + [1] * 9, prefix_edges, max_size=3)
    assert summaries[0]["size_ge2_positive"] > 0, summaries[0]
    assert summaries[0]["best_candidate_size"] >= 1, summaries[0]

    # Two positive singleton candidates exchange across one cut edge.  The
    # individual gains sum to two, but the exact batch gain is zero, so the
    # trial must be rejected and the state must remain unchanged.
    _, output, summaries = run_case(
        40, 2, [0] * 20 + [1] * 20, [(0, 20)])
    assert summaries[0]["positive_candidates"] >= 2, summaries[0]
    assert summaries[0]["accepted"] == 0, summaries[0]
    assert summaries[0]["batch_gain"] == 0, summaries[0]
    assert output == [0] * 20 + [1] * 20

    # The pair is profitable but the target has room for only one point; the
    # conservative incoming-only capacity rule must reject it.
    dense_target = [(u, v) for u in range(19, 40) for v in range(u + 1, 40)]
    dense_target += [(0, 1), (0, 19), (1, 20)]
    _, output, summaries = run_case(
        40, 2, [0] * 19 + [1] * 21, dense_target, max_size=4)
    assert summaries[0]["capacity_eliminated"] > 0, summaries[0]
    assert summaries[0]["accepted"] == 0, summaries[0]
    assert output == [0] * 19 + [1] * 21

    # Self-loops are present but must not affect block gains or the cut.
    self_loop_edges = [(0, 0), (1, 1)] + barrier_edges
    run_case(20, 2, [0] * 11 + [1] * 9, self_loop_edges)

    # Random small graphs exercise k=2 and k=4, including the exact verifier.
    for seed in range(8):
        rng = random.Random(seed)
        k = 2 if seed % 2 == 0 else 4
        n = 20 if k == 2 else 24
        labels = [vertex % k for vertex in range(n)]
        rng.shuffle(labels)
        edges = [(u, v) for u in range(n) for v in range(u, n)
                 if rng.random() < 0.12]
        run_case(n, k, labels, edges, max_size=6)

    # Disabled is the baseline path.  Compare it with an invocation where the
    # new block variables are absent.
    baseline = (20, 2, [0] * 11 + [1] * 9, barrier_edges)
    _, disabled_output, _ = run_case(*baseline, block=False, verify=False)
    environment_output = run_case(*baseline, block=None, verify=None)[1]
    assert disabled_output == environment_output
    print("PASS: block growth, negative seeds, best prefixes, exact batch rejection, "
          "capacity, self-loops, random verification, and disabled baseline")


if __name__ == "__main__":
    main()
