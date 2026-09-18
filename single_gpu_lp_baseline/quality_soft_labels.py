"""GPU continuous-label diffusion with upper-mass dual projection.

Candidate generator only: final discrete capacity is enforced by the existing
original-graph GPU search. No Jet labels or contracted graph are used.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument("dataset");p.add_argument("k",type=int);p.add_argument("initial",type=Path);p.add_argument("--output",type=Path,required=True)
p.add_argument("--powers",default="0,0.5,1");p.add_argument("--temperature-start",type=float,default=0.5);p.add_argument("--temperature-end",type=float,default=0.02);p.add_argument("--rounds",type=int,default=100)
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
if args.rounds<2 or args.temperature_start<=0 or args.temperature_end<=0:raise ValueError("positive temperatures and at least two rounds required")
root=Path(__file__).resolve().parent;data=root.parent/"dataset/Gpartition_dataset/Sym_CSR"/args.dataset
ptrpath=data/f"{args.dataset}_sym_indptr.bin";idxpath=data/f"{args.dataset}_sym_indices.bin"
ptr_np=np.fromfile(ptrpath,dtype="<i8");idx_np=np.fromfile(idxpath,dtype="<i8");n=len(ptr_np)-1
labels_np=np.fromfile(args.initial,dtype="<i4");assert len(labels_np)==n
device="cuda";ptr=torch.from_numpy(ptr_np).to(device);idx=torch.from_numpy(idx_np).to(device)
degree=(ptr[1:]-ptr[:-1]).float();initial=torch.from_numpy(labels_np.astype(np.int64)).to(device)
adj=torch.sparse_csr_tensor(ptr,idx,torch.ones(len(idx_np),device=device),size=(n,n),device=device,check_invariants=True)
del idx_np
cap=np.floor(n/args.k*1.1);average_degree=float(degree.mean())
records=[]
with torch.no_grad():
    for power in map(float,args.powers.split(",")):
        torch.cuda.synchronize();start=time.monotonic()
        x=torch.nn.functional.one_hot(initial,num_classes=args.k).float()*0.9+0.1/args.k
        dual=torch.zeros(args.k,device=device);normalization=degree.clamp_min(1).pow(-power).unsqueeze(1)
        scale=average_degree**(1-power)
        for iteration in range(args.rounds):
            temperature=scale*(args.temperature_start*(args.temperature_end/args.temperature_start)**(iteration/(args.rounds-1)))
            score=torch.sparse.mm(adj,x)*normalization
            for _ in range(8):
                prob=torch.softmax((score-dual)/temperature,dim=1)
                mass=prob.sum(0)
                dual=(dual+temperature*torch.log((mass/cap).clamp_min(1e-8))).clamp_min(0)
            x=0.5*x+0.5*prob
        result=x.argmax(1);result=torch.where(degree==0,initial,result)
        torch.cuda.synchronize();seconds=time.monotonic()-start
        tag=f"p{power}";candidate=args.output/f"{tag}.candidate.parts"
        result.cpu().numpy().astype("<i4").tofile(candidate)
        env=os.environ.copy();env.update(INITIAL_PARTITION=str(candidate.resolve()),FIELD_ROUNDS="32",GLOBAL_CYCLES="10",RESTORE_BEST_CYCLE="0",INCREMENTAL_CUT_VERIFY="1")
        with (args.output/f"{tag}.log").open("w") as log:
            run=subprocess.run([str(root/"build-gh200/single_gpu_lp_baseline"),str(ptrpath),str(idxpath),str(args.k),str(args.output/f"{tag}.parts"),"30","50","1","1.10"],env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(power=power,temperature_start=args.temperature_start,temperature_end=args.temperature_end,rounds=args.rounds,soft_seconds=seconds,exit_code=run.returncode)
        if run.returncode==0:
            text=(args.output/f"{tag}.log").read_text();final=dict(re.findall(r"(\w+)=([^\s]+)",next(x for x in text.splitlines() if x.startswith("final_cut="))))
            assert final["feasible"]=="1"
            record.update(cut=int(final["final_cut"])//2,ratio=int(final["final_cut"])/(ptr_np[-1]),refine_seconds=float(re.search(r"partition_seconds=(\S+)",text).group(1)))
        records.append(record);print(record,flush=True)
        (args.output/"summary.json").write_text(json.dumps(records,indent=2)+"\n")
