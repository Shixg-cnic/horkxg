"""Select and independently validate feasible regional results; exclude loose states."""
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parent
out=root.parent/"results/single_gpu_lp/quality_community"
before={(r["dataset"],r["k"]):r for r in csv.DictReader((root.parent/"results/single_gpu_lp/quality_20260908_v2/best_observed.csv").open())}
results=[]
for row in csv.DictReader((out/"region_benchmark/summary.csv").open()):
    old=before[(row["dataset"],row["k"])];chosen=dict(row)
    if row["dataset"]=="products" and row["k"]=="8":
        for tag in ["products_k8_region4096","products_k8_region65536","products_k8_lagrange","products_k8_residual"]:
            path=out/f"{tag}.log"
            if not path.exists():continue
            match=re.search(r"^final_cut=.*$",path.read_text(),re.M)
            if not match:continue
            metrics=dict(re.findall(r"(\w+)=([^\s]+)",match.group()))
            if int(metrics["final_cut"])<int(chosen["edge_cut"]):chosen.update(edge_cut=metrics["final_cut"],cut_ratio=metrics["cut_ratio"],cpu_seconds=metrics["region_seconds"],parts=str(path.with_suffix(".parts")))
    chosen.update(previous_ratio=old["cut_ratio"],jet_ratio=old["jet_ratio"],jet_cut=old["jet_cut"],gap_to_jet=int(chosen["edge_cut"])/int(old["jet_cut"])-1)
    data=root.parent/"dataset/Gpartition_dataset/Sym_CSR"/row["dataset"]
    run=subprocess.run([sys.executable,str(root/"verify_quality.py"),str(data),row["k"],chosen["parts"]],capture_output=True,text=True,check=True)
    proof=json.loads(run.stdout);assert proof["edge_cut"]==int(chosen["edge_cut"])
    chosen["cut_ratio"]=proof["cut_ratio"];chosen["vertex_imbalance"]=proof["vertex_imbalance"]
    chosen["validation"]=proof;results.append(chosen)
    print(row["dataset"],row["k"],chosen["edge_cut"],chosen["cut_ratio"],"verified",flush=True)
results.sort(key=lambda r:(r["dataset"],int(r["k"])))
with (out/"best_observed.csv").open("w") as f:
    writer=csv.DictWriter(f,fieldnames=[k for k in results[0] if k!="validation"]);writer.writeheader()
    for r in results:writer.writerow({k:v for k,v in r.items() if k!="validation"})
(out/"validated_results.json").write_text(json.dumps(results,indent=2)+"\n")
