"""Report best observed feasible results, with candidate provenance and Jet gaps."""
import csv
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parent
prior=root.parent/"results/single_gpu_lp/quality_20260908"
current=root.parent/"results/single_gpu_lp/quality_20260908_v2"
jet={(r["dataset"],int(r["k"])):r for r in csv.DictReader((root.parent/"results/Jet/sweep_imb110/summary.csv").open()) if r["status"]=="ok"}
all_results=[]
for dataset in ["products","com-LiveJournal"]:
    for k in [2,4,8,16,32]:
        candidates=[]
        def cpu_log(path,method):
            if not path.exists():return
            text=path.read_text();match=re.search(r"^final_cut=.*$",text,re.M)
            if not match:return
            final=dict(re.findall(r"(\w+)=([^\s]+)",match.group()))
            parts=path.with_suffix(".parts")
            if not parts.exists():return
            cut=int(final["final_cut"])
            if "edge_entries" in final:cut//=2
            candidates.append(dict(method=method,edge_cut=cut,parts=str(parts.resolve())))
        initial=prior/"benchmark"/f"{dataset}_k{k}_r0"
        if not initial.exists():initial=current/f"{dataset}_k{k}_gpu"
        if (initial/"metrics.json").exists():
            m=json.loads((initial/"metrics.json").read_text());candidates.append(dict(method="four_start_gpu",edge_cut=m["edge_cut"],parts=str((initial/"best.parts").resolve())))
        for log in current.glob(f"{dataset}_k{k}_*.log"):cpu_log(log,log.stem)
        cpu_log(current/"deep"/f"{dataset}_k{k}.log","deeper_gpu_then_groups")
        for log in (current/"resume").glob(f"{dataset}_k{k}_*.log"):cpu_log(log,"groups_then_gpu_"+log.stem)
        for log in (current/"geometry").glob(f"{dataset}_k{k}_*.log"):cpu_log(log,"seed_geometry_"+log.stem)
        if dataset=="products" and k==4:candidates.append(dict(method="previous_best",edge_cut=2154026,parts=str((prior/"best_pipeline/best.parts").resolve())))
        if dataset=="products":
            path=current/"bisect_relaxed_products"/f"k{k}_refine.log"
            if path.exists():
                match=re.search(r"^final_cut=(\d+)",path.read_text(),re.M)
                if match:candidates.append(dict(method="recursive_initialization_then_gpu",edge_cut=int(match.group(1))//2,parts=str((path.parent/f"k{k}_refined.parts").resolve())))
        best=min(candidates,key=lambda c:c["edge_cut"])
        j=jet[(dataset,k)];m=int(j["edges"])
        all_results.append(dict(dataset=dataset,k=k,edge_cut=best["edge_cut"],cut_ratio=best["edge_cut"]/m,jet_cut=int(j["edge_cut"]),jet_ratio=float(j["edge_cut_ratio"]),gap_to_jet=best["edge_cut"]/int(j["edge_cut"])-1,method=best["method"],parts=best["parts"],candidates=candidates))
with (current/"best_observed.csv").open("w") as f:
    writer=csv.DictWriter(f,fieldnames=[k for k in all_results[0] if k!="candidates"]);writer.writeheader()
    for r in all_results:writer.writerow({k:v for k,v in r.items() if k!="candidates"})
(current/"selection.json").write_text(json.dumps(all_results,indent=2)+"\n")
for r in all_results:print(r["dataset"],r["k"],r["edge_cut"],f'{r["cut_ratio"]:.8f}',f'Jet gap {100*r["gap_to_jet"]:.2f}%',r["method"])
