"""Uniform seed-geometry ablation on both graphs at every partition count."""
import csv
import os
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parent
out=root.parent/"results/single_gpu_lp/quality_20260908_v2/geometry"
out.mkdir(parents=True,exist_ok=False)
with (out/"summary.csv").open("w") as f:
    writer=csv.writer(f);writer.writerow(["dataset","k","distance_power","edge_cut","cut_ratio","gpu_seconds","parts"])
    for dataset in ["products","com-LiveJournal"]:
        for k in [2,4,8,16,32]:
            for power in [2,3]:
                dest=out/f"{dataset}_k{k}_p{power}";data=root.parent/"dataset/Gpartition_dataset/Sym_CSR"/dataset
                env=os.environ.copy();env.pop("INITIAL_PARTITION",None)
                env.update(SEARCH_SEED="2",SEED_DISTANCE_POWER=str(power),FIELD_ROUNDS="32",GLOBAL_CYCLES="10",RESTORE_BEST_CYCLE="0",INCREMENTAL_CUT_VERIFY="1")
                with dest.with_suffix(".log").open("w") as log:
                    run=subprocess.run([str(root/"build-gh200/single_gpu_lp_baseline"),str(data/f"{dataset}_sym_indptr.bin"),str(data/f"{dataset}_sym_indices.bin"),str(k),str(dest.with_suffix(".parts")),"30","50","1","1.10"],env=env,stdout=log,stderr=subprocess.STDOUT)
                run.check_returncode();text=dest.with_suffix(".log").read_text()
                final=dict(re.findall(r"(\w+)=([^\s]+)",next(x for x in text.splitlines() if x.startswith("final_cut="))))
                assert final["feasible"]=="1"
                row=[dataset,k,power,int(final["final_cut"])//2,final["cut_ratio"],re.search(r"partition_seconds=(\S+)",text).group(1),str(dest.with_suffix(".parts"))]
                writer.writerow(row);f.flush();print(row,flush=True)
