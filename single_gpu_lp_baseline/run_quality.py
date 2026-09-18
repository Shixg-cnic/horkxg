"""Single-level multi-start GPU search with optional CPU reference refinement."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("k", type=int)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seeds", default="0,1,2,3")
parser.add_argument("--field-rounds", type=int, default=32)
parser.add_argument("--cycles", type=int, default=10)
parser.add_argument("--reference-refine", action="store_true")
parser.add_argument("--group-exchange", action="store_true")
parser.add_argument("--group-redirect", action="store_true")
parser.add_argument("--distance-power", type=int, choices=[1,2,3,4], default=1)
parser.add_argument("--initial-partition", type=Path)
parser.add_argument("--verify", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
args.output.mkdir(parents=True, exist_ok=False)
data = root.parent / "dataset/Gpartition_dataset/Sym_CSR" / args.dataset
ptr = data / f"{args.dataset}_sym_indptr.bin"
idx = data / f"{args.dataset}_sym_indices.bin"
records = []
start = time.monotonic()
for seed in map(int, args.seeds.split(",")):
    parts = args.output / f"seed{seed}.parts"
    env = os.environ.copy()
    env["SEED_DISTANCE_POWER"] = str(args.distance_power)
    if args.initial_partition:env["INITIAL_PARTITION"] = str(args.initial_partition.resolve())
    else:env.pop("INITIAL_PARTITION",None)
    env.update(SEARCH_SEED=str(seed), FIELD_ROUNDS=str(args.field_rounds), GLOBAL_CYCLES=str(args.cycles), RESTORE_BEST_CYCLE="0", INCREMENTAL_CUT_VERIFY=str(int(args.verify)))
    command = [str(root / "build-gh200/single_gpu_lp_baseline"), str(ptr), str(idx), str(args.k), str(parts), "30", "50", "1", "1.10"]
    run = subprocess.run(command, env=env, text=True, capture_output=True)
    (args.output / f"seed{seed}.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    final = dict(re.findall(r"(\w+)=([^\s]+)", next(x for x in run.stdout.splitlines() if x.startswith("final_cut="))))
    if final["feasible"] != "1":
        raise RuntimeError("infeasible output")
    records.append(dict(seed=seed, directed_cut=int(final["final_cut"]), edge_entries=int(final["edge_entries"]), seconds=float(re.search(r"partition_seconds=(\S+)", run.stdout).group(1)), parts=str(parts.resolve())))
best = min(records, key=lambda r:r["directed_cut"])
result = dict(dataset=args.dataset, k=args.k, max_vertex_ratio=1.10, field_rounds=args.field_rounds, cycles=args.cycles, distance_power=args.distance_power, initial_partition=str(args.initial_partition) if args.initial_partition else None, verification=args.verify, candidates=records, best_seed=best["seed"], edge_cut=best["directed_cut"]//2, edge_cut_ratio=best["directed_cut"]/best["edge_entries"], gpu_search_seconds=sum(r["seconds"] for r in records))
if args.reference_refine:
    command = [str(root / "build-gh200/quality_refine"),str(ptr),str(idx),best["parts"],str(args.output / "best.parts"),str(args.k),"1.10","3","20000","42","1","256"]
    reference_env=os.environ.copy();reference_env.update(GROUP_EXCHANGE=str(int(args.group_exchange)),GROUP_REDIRECT=str(int(args.group_redirect)),GROUP_LOSS_LIMIT="256")
    run = subprocess.run(command, env=reference_env, capture_output=True, text=True)
    (args.output / "refine.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    final = dict(re.findall(r"(\w+)=([^\s]+)", next(x for x in run.stdout.splitlines() if x.startswith("final_cut="))))
    result.update(edge_cut=int(final["final_cut"]), edge_cut_ratio=2*int(final["final_cut"])/best["edge_entries"], cpu_refine_seconds=float(final["refine_seconds"]))
    result.update(group_exchange=args.group_exchange,group_redirect=args.group_redirect)
else:
    shutil.copyfile(best["parts"], args.output / "best.parts")
result["total_wall_seconds"] = time.monotonic()-start
(args.output / "metrics.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
