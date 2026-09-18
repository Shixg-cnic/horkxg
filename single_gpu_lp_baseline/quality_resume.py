"""Re-enter GPU field search from improved CPU labels, retaining feasible best."""
import csv
import os
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parent
old=root.parent/"results/single_gpu_lp/quality_20260908"
current=root.parent/"results/single_gpu_lp/quality_20260908_v2"
out=current/"resume";out.mkdir(parents=True,exist_ok=False)
with (out/"summary.csv").open("w") as f:
    writer=csv.writer(f);writer.writerow(["dataset","k","field","input","edge_cut","cut_ratio","gpu_seconds","parts"])
    for k in [2,4,8,16,32]:
        candidates=[]
        for log in list(current.glob(f"products_k{k}_exchange*.log"))+list((current/"deep").glob(f"products_k{k}.log")):
            text=log.read_text();match=re.search(r"final_cut=(\d+)",text)
            if match and log.with_suffix(".parts").exists():candidates.append((int(match.group(1)),log.with_suffix(".parts")))
        if k==4:candidates.append((2154026,old/"best_pipeline/best.parts"))
        _,initial=min(candidates)
        for field in [8,32]:
            dest=out/f"products_k{k}_f{field}";env=os.environ.copy();env.update(INITIAL_PARTITION=str(initial.resolve()),FIELD_ROUNDS=str(field),GLOBAL_CYCLES="10",RESTORE_BEST_CYCLE="0",INCREMENTAL_CUT_VERIFY="1")
            data=root.parent/"dataset/Gpartition_dataset/Sym_CSR/products"
            command=[str(root/"build-gh200/single_gpu_lp_baseline"),str(data/"products_sym_indptr.bin"),str(data/"products_sym_indices.bin"),str(k),str(dest.with_suffix(".parts")),"30","50","1","1.10"]
            with dest.with_suffix(".log").open("w") as log:run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            run.check_returncode();text=dest.with_suffix(".log").read_text()
            final=dict(re.findall(r"(\w+)=([^\s]+)",next(x for x in text.splitlines() if x.startswith("final_cut="))))
            assert final["feasible"]=="1"
            row=["products",k,field,str(initial),int(final["final_cut"])//2,final["cut_ratio"],re.search(r"partition_seconds=(\S+)",text).group(1),str(dest.with_suffix(".parts"))]
            writer.writerow(row);f.flush();print(row,flush=True)
