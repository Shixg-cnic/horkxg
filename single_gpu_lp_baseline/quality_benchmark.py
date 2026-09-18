"""Compare fixed four-start policy with baseline; include every start in timing."""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parent
out=root.parent / "results/single_gpu_lp/quality_20260908/benchmark"
out.mkdir(parents=True,exist_ok=False)
with (out/"summary.csv").open("w") as f:
    writer=csv.writer(f)
    writer.writerow(["dataset","k","repeat","method","edge_cut","cut_ratio","gpu_seconds"])
    for dataset,k,repeat in [("products",k,0) for k in [2,4,8,16,32]]+[("products",4,r) for r in [1,2]]+[("com-LiveJournal",4,0)]:
        tag=f"{dataset}_k{k}_r{repeat}"
        data=root.parent/"dataset/Gpartition_dataset/Sym_CSR"/dataset
        env=os.environ.copy()
        env.update(SEARCH_SEED="0",FIELD_ROUNDS="8",GLOBAL_CYCLES="5",RESTORE_BEST_CYCLE="0",INCREMENTAL_CUT_VERIFY="0")
        run=subprocess.run([str(root/"build-gh200/single_gpu_lp_baseline"),str(data/f"{dataset}_sym_indptr.bin"),str(data/f"{dataset}_sym_indices.bin"),str(k),str(out/f"{tag}.baseline.parts"),"30","50","1","1.10"],env=env,text=True,capture_output=True)
        (out/f"{tag}.baseline.log").write_text(run.stdout+run.stderr);run.check_returncode()
        final=dict(re.findall(r"(\w+)=([^\s]+)",next(x for x in run.stdout.splitlines() if x.startswith("final_cut="))))
        assert final["feasible"]=="1"
        writer.writerow([dataset,k,repeat,"baseline",int(final["final_cut"])//2,final["cut_ratio"],re.search(r"partition_seconds=(\S+)",run.stdout).group(1)])
        subprocess.run([sys.executable,str(root/"run_quality.py"),dataset,str(k),"--output",str(out/tag)],check=True,stdout=subprocess.DEVNULL)
        metrics=json.loads((out/tag/"metrics.json").read_text())
        writer.writerow([dataset,k,repeat,"gpu_multistart",metrics["edge_cut"],metrics["edge_cut_ratio"],metrics["gpu_search_seconds"]]);f.flush()
        print(tag,final["cut_ratio"],metrics["edge_cut_ratio"],flush=True)
