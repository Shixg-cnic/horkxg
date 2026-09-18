"""Independently verify each selected multi-k result against full original CSR."""
import csv
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
out=root.parent/"results/single_gpu_lp/quality_20260908_v2"
with (out/"independent_validation.jsonl").open("w") as f:
    for row in csv.DictReader((out/"best_observed.csv").open()):
        data=root.parent/"dataset/Gpartition_dataset/Sym_CSR"/row["dataset"]
        run=subprocess.run([sys.executable,str(root/"verify_quality.py"),str(data),row["k"],row["parts"]],capture_output=True,text=True,check=True)
        result=json.loads(run.stdout)
        assert result["edge_cut"]==int(row["edge_cut"])
        result.update(dataset=row["dataset"],k=int(row["k"]))
        f.write(json.dumps(result)+"\n");f.flush()
        print(row["dataset"],row["k"],"verified",result["edge_cut"],flush=True)
