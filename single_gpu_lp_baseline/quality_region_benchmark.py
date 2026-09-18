"""Uniform regional-search policy over both graphs and all k."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import os
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parent
previous=root.parent/"results/single_gpu_lp/quality_20260908_v2/best_observed.csv"
out=root.parent/"results/single_gpu_lp/quality_community/region_benchmark";out.mkdir(parents=True,exist_ok=False)
def work(row):
    dataset=row["dataset"];k=row["k"];data=root.parent/"dataset/Gpartition_dataset/Sym_CSR"/dataset
    dest=out/f"{dataset}_k{k}";env=os.environ.copy();env.update(REGION_LAGRANGE_STEPS="0",REGION_SECONDS="120")
    cmd=[str(root/"build-gh200/region_refine"),str(data/f"{dataset}_sym_indptr.bin"),str(data/f"{dataset}_sym_indices.bin"),row["parts"],str(dest.with_suffix(".parts")),k,"65536","4","42"]
    with dest.with_suffix(".log").open("w") as log:run=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=150)
    run.check_returncode();text=dest.with_suffix(".log").read_text();final=dict(re.findall(r"(\w+)=([^\s]+)",next(x for x in text.splitlines() if x.startswith("final_cut="))))
    assert int(final["final_cut"])<=int(row["edge_cut"])
    result=[dataset,k,row["edge_cut"],final["final_cut"],final["cut_ratio"],final["region_seconds"],str(dest.with_suffix(".parts"))]
    print(result,flush=True);return result
with ThreadPoolExecutor(max_workers=2) as pool,(out/"summary.csv").open("w") as f:
    writer=csv.writer(f);writer.writerow(["dataset","k","initial_cut","edge_cut","cut_ratio","cpu_seconds","parts"])
    tasks=[pool.submit(work,row) for row in csv.DictReader(previous.open())]
    for task in as_completed(tasks):writer.writerow(task.result());f.flush()
