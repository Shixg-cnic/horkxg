"""Multi-k group-capacity ablation; both variants start from identical labels."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parent
previous=root.parent/"results/single_gpu_lp/quality_20260908"
out=root.parent/"results/single_gpu_lp/quality_20260908_v2"
out.mkdir(parents=True,exist_ok=False)

def refine(dataset,k,initial,exchange,tag):
    data=root.parent/"dataset/Gpartition_dataset/Sym_CSR"/dataset
    dest=out/f"{tag}_exchange{exchange}"
    env=os.environ.copy();env.update(GROUP_EXCHANGE=str(exchange),GROUP_LOSS_LIMIT="256")
    cmd=[str(root/"build-gh200/quality_refine"),str(data/f"{dataset}_sym_indptr.bin"),str(data/f"{dataset}_sym_indices.bin"),str(initial),str(dest.with_suffix(".parts")),str(k),"1.10","2","10000","42","1","128"]
    try:
        with dest.with_suffix(".log").open("w") as log:
            run=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        if run.returncode:raise RuntimeError(f"exit={run.returncode}")
        text=dest.with_suffix(".log").read_text()
        final=dict(re.findall(r"(\w+)=([^\s]+)",next(x for x in text.splitlines() if x.startswith("final_cut="))))
        initial_cut=int(re.search(r"initial_cut=(\d+)",text).group(1))
        row=[dataset,k,tag,exchange,"ok",initial_cut,final["final_cut"],final["cut_ratio"],final["refine_seconds"],final["vertex_imb"],str(dest.with_suffix(".parts"))]
    except (subprocess.TimeoutExpired,RuntimeError) as error:
        row=[dataset,k,tag,exchange,str(error),"","","","","",""]
    print(row,flush=True)
    return row

with ThreadPoolExecutor(max_workers=2) as executor, (out/"summary.csv").open("w") as f:
    writer=csv.writer(f);writer.writerow(["dataset","k","tag","exchange","status","initial_cut","edge_cut","cut_ratio","cpu_seconds","vertex_imb","parts"])
    pending=[]
    for dataset in ["products","com-LiveJournal"]:
        for k in [2,4,8,16,32]:
            tag=f"{dataset}_k{k}"
            old=previous/"benchmark"/f"{dataset}_k{k}_r0"
            if old.is_dir():
                initial=old/"best.parts"
            else:
                gpu=out/f"{tag}_gpu"
                with (out/f"{tag}_gpu.log").open("w") as log:
                    subprocess.run([sys.executable,str(root/"run_quality.py"),dataset,str(k),"--output",str(gpu)],stdout=log,stderr=subprocess.STDOUT,check=True)
                initial=gpu/"best.parts"
            for exchange in [0,1]:pending.append(executor.submit(refine,dataset,k,initial,exchange,tag))
    for exchange in [0,1]:pending.append(executor.submit(refine,"products",4,previous/"best_pipeline/best.parts",exchange,"products_k4_previous_best"))
    for task in as_completed(pending):
        writer.writerow(task.result());f.flush()
