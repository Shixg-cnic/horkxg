"""Reproducible single-level initialization ablation; never consumes Jet labels."""
import csv
import itertools
import os
from pathlib import Path
import re
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["init", "seed", "deep"], default="init")
parser.add_argument("--output", type=Path)
args = parser.parse_args()

root = Path(__file__).resolve().parent
out = args.output or root.parent / f"results/single_gpu_lp/quality_20260908/{args.mode}_sweep"
out.mkdir(parents=True, exist_ok=False)
data = root.parent / "dataset/Gpartition_dataset/Sym_CSR/products"
with (out / "summary.csv").open("w") as stream:
    writer = csv.writer(stream)
    writer.writerow(["seeds_per_part", "field_rounds", "cycles", "cut_ratio", "directed_cut", "seconds", "feasible", "parts", "search_seed", "restore"])
    configs = [(s,f,c,0,0) for s,f,c in itertools.product([1,2,4,8],[4,8,16,32],[5,10])] if args.mode == "init" else [(1,32,10,seed,restore) for seed,restore in itertools.product(range(17),[0,1])]
    if args.mode == "deep":
        configs = [(1,f,c,seed,0) for seed,f,c in itertools.product([0,2],[64,128],[10,20,40])]
    for seeds, field, cycles, search_seed, restore in configs:
        tag = f"s{seeds}_f{field}_c{cycles}_r{search_seed}_restore{restore}"
        parts = out / f"{tag}.parts"
        env = os.environ.copy()
        env.update(FIELD_ROUNDS=str(field), GLOBAL_CYCLES=str(cycles), INCREMENTAL_CUT_VERIFY="1", SEARCH_SEED=str(search_seed), RESTORE_BEST_CYCLE=str(restore))
        cmd = [str(root / "build-gh200/single_gpu_lp_baseline"), str(data / "products_sym_indptr.bin"), str(data / "products_sym_indices.bin"), "4", str(parts), "30", "50", str(seeds), "1.10"]
        run = subprocess.run(cmd, env=env, capture_output=True, text=True)
        (out / f"{tag}.log").write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f"{tag}: {run.stderr}")
        final = dict(re.findall(r"(\w+)=([^\s]+)", next(x for x in run.stdout.splitlines() if x.startswith("final_cut="))))
        seconds = re.search(r"partition_seconds=(\S+)", run.stdout).group(1)
        writer.writerow([seeds, field, cycles, final["cut_ratio"], final["final_cut"], seconds, final["feasible"], parts, search_seed, restore])
        stream.flush()
        print(tag, final["cut_ratio"], seconds, flush=True)
