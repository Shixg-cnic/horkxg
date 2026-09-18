"""Correctness and control-flow checks for the ENABLE_FIELD switch."""

from array import array
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
BINARY = Path(os.environ.get(
    "FIELD_TEST_BINARY", ROOT / "build-gh200/single_gpu_lp_baseline"))


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


def run_case(enable_field, global_cycles=1, time_budget=None):
    n = 12
    labels = [0] * 6 + [1] * 6
    edges = [(index, (index + 1) % n) for index in range(n)]
    edges += [(0, 6), (1, 7), (2, 8), (3, 9)]
    offsets, indices = make_csr(n, edges)
    with tempfile.TemporaryDirectory(prefix="cuda-field-test-") as directory:
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
            FIELD_ROUNDS="2",
            GLOBAL_CYCLES=str(global_cycles),
            BALANCE_ROUNDS="0",
            POLISH_ROUNDS="0",
            PAIR_EXCHANGE="0",
            BLOCK_LP="0",
        )
        if enable_field is None:
            environment.pop("ENABLE_FIELD", None)
        else:
            environment["ENABLE_FIELD"] = str(enable_field)
        if time_budget is None:
            environment.pop("TIME_BUDGET_SECONDS", None)
        else:
            environment["TIME_BUDGET_SECONDS"] = str(time_budget)
        result = subprocess.run(
            [str(BINARY), str(root / "ptr"), str(root / "idx"), "2",
             str(root / "out"), "30", "2", "1", "1.10"],
            env=environment, capture_output=True, text=True, check=True,
        )
        output = array("i")
        with (root / "out").open("rb") as stream:
            output.fromfile(stream, n)
        return result.stdout, output.tolist()


def main():
    if not BINARY.exists():
        raise SystemExit(f"missing CUDA binary: {BINARY}")

    default_stdout, default_output = run_case(None)
    enabled_stdout, enabled_output = run_case(1)
    disabled_stdout, disabled_output = run_case(0)
    budget_stdout, budget_output = run_case(0, global_cycles=3,
                                            time_budget=0.0001)

    assert default_output == enabled_output
    assert "enable_field=1" in enabled_stdout
    assert "label_field=" in enabled_stdout
    assert "stage=field" in enabled_stdout
    assert re.search(r"trial=0 .*field_applied=1", enabled_stdout)

    assert disabled_output != []
    assert all(label in (0, 1) for label in disabled_output)
    assert "enable_field=0" in disabled_stdout
    assert not re.search(r"^label_field=", disabled_stdout, re.MULTILINE)
    assert not re.search(r"^  field=", disabled_stdout, re.MULTILINE)
    assert not re.search(r"stage=field", disabled_stdout)
    assert re.search(r"trial=0 .*field_applied=0", disabled_stdout)
    assert "time_budget_summary enabled=0" in disabled_stdout

    assert budget_output != []
    assert "time_budget_stop" in budget_stdout
    assert "completed_global_cycles=0" in budget_stdout
    assert "restored_to_cycle_start=1" in budget_stdout
    assert "time_budget_summary enabled=1 stopped=1" in budget_stdout

    print("PASS: ENABLE_FIELD default/on equivalence and field-free execution path")


if __name__ == "__main__":
    main()
