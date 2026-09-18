"""Apply one deeper-search policy to both datasets and every k, not only k=4."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
old=root.parent/"results/single_gpu_lp/quality_20260908/benchmark"
current=root.parent/"results/single_gpu_lp/quality_20260908_v2"
out=current/"deep"
out.mkdir(parents=True,exist_ok=False)

def refine(dataset,k,path):
    dest=out/f"{dataset}_k{k}"
    data=root.parent/"dataset/Gpartition_dataset/Sym_CSR"/dataset
    env=os.environ.copy();env.update(GROUP_EXCHANGE="1",GROUP_LOSS_LIMIT="256")
    command=[str(root/"build-gh200/quality_refine"),str(data/f"{dataset}_sym_indptr.bin"),str(data/f"{dataset}_sym_indices.bin"),str(path),str(dest.with_suffix(".parts")),str(k),"1.10","2","10000","42","1","128"]
    with dest.with_suffix(".log").open("w") as log:
        run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
    run.check_returncode()
    text=dest.with_suffix(".log").read_text()
    final=dict(re.findall(r"(\w+)=([^\s]+)",next(x for x in text.splitlines() if x.startswith("final_cut="))))
    return [dataset,k,final["final_cut"],final["cut_ratio"],final["refine_seconds"],str(dest.with_suffix(".parts"))]

# Wait for the first suite's GPU jobs so GPU timings are not concurrent.
deadline=time.monotonic()+180
while not (current/"com-LiveJournal_k32_gpu/metrics.json").exists():
    if time.monotonic()>deadline:raise TimeoutError("missing initial GPU results")
    time.sleep(1)
with ThreadPoolExecutor(max_workers=2) as executor,(out/"summary.csv").open("w") as f:
    writer=csv.writer(f);writer.writerow(["dataset","k","edge_cut","cut_ratio","cpu_seconds","parts"])
    pending=[]
    for dataset in ["products","com-LiveJournal"]:
        for k in [2,4,8,16,32]:
            initial=old/f"{dataset}_k{k}_r0"
            if not initial.exists():initial=current/f"{dataset}_k{k}_gpu"
            metrics=json.loads((initial/"metrics.json").read_text())
            seed=metrics["best_seed"]
            gpu=out/f"{dataset}_k{k}_gpu"
            with (out/f"{dataset}_k{k}_gpu.log").open("w") as log:
                subprocess.run([sys.executable,str(root/"run_quality.py"),dataset,str(k),"--seeds",str(seed),"--field-rounds","64","--cycles","40","--output",str(gpu)],stdout=log,stderr=subprocess.STDOUT,check=True)
            fresh=json.loads((gpu/"metrics.json").read_text())
            print(dataset,k,"old",metrics["edge_cut_ratio"],"deep",fresh["edge_cut_ratio"],flush=True)
            pending.append(executor.submit(refine,dataset,k,gpu/"best.parts"))
    for task in as_completed(pending):writer.writerow(task.result());f.flush()
