"""Experimental recursive LP initialization: no vertex contraction.

Default intermediate splits are near-exactly balanced. With --split-ratio 1.10,
intermediate imbalance can accumulate; only feasible globally refined outputs
may be used as final partitions. The tree provides k=2,4,8,16,32 candidates.
Temporary child CSRs keep original vertices, relabeled locally, and internal edges.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np

p=argparse.ArgumentParser();p.add_argument("dataset");p.add_argument("--output",type=Path,required=True)
p.add_argument("--split-ratio",type=float,default=1.0);p.add_argument("--global-refine",action="store_true")
args=p.parse_args();root=Path(__file__).resolve().parent
args.output.mkdir(parents=True,exist_ok=False)
data=root.parent/"dataset/Gpartition_dataset/Sym_CSR"/args.dataset
ptr=data/f"{args.dataset}_sym_indptr.bin";idx=data/f"{args.dataset}_sym_indices.bin"
n=ptr.stat().st_size//8-1
nodes=[(ptr,idx,np.arange(n,dtype=np.int64))];labels=np.zeros(n,dtype=np.int32)
total_gpu=0.;start=time.monotonic()
def child_csr(ptrpath,idxpath,part,child,path):
    offsets=np.memmap(ptrpath,dtype="<i8",mode="r");indices=np.memmap(idxpath,dtype="<i8",mode="r")
    vertices=np.flatnonzero(part==child);mapping=np.full(len(part),-1,dtype=np.int64);mapping[vertices]=np.arange(len(vertices))
    degrees=np.zeros(len(part),dtype=np.int64)
    with (path/"indices.bin").open("wb") as stream:
        for first in range(0,len(part),4096):
            last=min(len(part),first+4096)
            src=np.repeat(np.arange(last-first),np.diff(offsets[first:last+1]))
            nbr=indices[offsets[first]:offsets[last]]
            keep=(part[first:last][src]==child)&(part[nbr]==child)
            degrees[first:last]=np.bincount(src[keep],minlength=last-first)
            mapping[nbr[keep]].astype("<i8").tofile(stream)
    np.r_[np.int64(0),np.cumsum(degrees[vertices])].astype("<i8").tofile(path/"indptr.bin")
    return vertices

with (args.output/"summary.jsonl").open("w") as summary:
    for depth in range(1,6):
        children=[]
        for node,(nodeptr,nodeidx,globalids) in enumerate(nodes):
            directory=args.output/f"d{depth}_n{node}";directory.mkdir()
            size=len(globalids);cap=max((size+1)//2,int(np.floor(size/2*args.split_ratio)))
            # Choose a float ratio whose capacity is exactly ceil(size/2).
            ratio=np.float32(2*cap/size)
            while int(np.floor(np.longdouble(size)/2*np.longdouble(ratio)))<cap:ratio=np.nextafter(ratio,np.float32(np.inf))
            env=os.environ.copy();env.pop("INITIAL_PARTITION",None);env.update(SEARCH_SEED="2",FIELD_ROUNDS="32",GLOBAL_CYCLES="10",RESTORE_BEST_CYCLE="0",INCREMENTAL_CUT_VERIFY="0")
            command=[str(root/"build-gh200/single_gpu_lp_baseline"),str(nodeptr),str(nodeidx),"2",str(directory/"split.parts"),"30","50","1",str(float(ratio))]
            with (directory/"run.log").open("w") as log:
                run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            run.check_returncode()
            text=(directory/"run.log").read_text();total_gpu+=float(re.search(r"partition_seconds=(\S+)",text).group(1))
            part=np.fromfile(directory/"split.parts",dtype="<i4")
            assert len(part)==size and set(np.unique(part))=={0,1}
            assert max(np.bincount(part))<=cap
            labels[globalids]=2*node+part
            if depth<5:
                for child in range(2):
                    dest=directory/f"child{child}";dest.mkdir()
                    local=child_csr(nodeptr,nodeidx,part,child,dest)
                    children.append((dest/"indptr.bin",dest/"indices.bin",globalids[local]))
        k=2**depth;labels.tofile(args.output/f"k{k}.parts")
        original_ptr=np.memmap(ptr,dtype="<i8",mode="r");original_idx=np.memmap(idx,dtype="<i8",mode="r")
        directed=0
        for first in range(0,n,8192):
            last=min(n,first+8192);src=np.repeat(labels[first:last],np.diff(original_ptr[first:last+1]))
            directed+=int(np.count_nonzero(src!=labels[original_idx[original_ptr[first]:original_ptr[last]]]))
        result=dict(dataset=args.dataset,k=k,edge_cut=directed//2,cut_ratio=directed/len(original_idx),cumulative_gpu_seconds=total_gpu,cumulative_wall_seconds=time.monotonic()-start)
        result["initial_vertex_imb"]=float(np.bincount(labels).max())*k/n
        result["initial_feasible"]=int(np.bincount(labels).max())<=int(np.floor(n/k*1.1))
        result["split_ratio"]=args.split_ratio
        if args.global_refine:
            env=os.environ.copy();env.update(INITIAL_PARTITION=str((args.output/f"k{k}.parts").resolve()),FIELD_ROUNDS="32",GLOBAL_CYCLES="10",RESTORE_BEST_CYCLE="0",INCREMENTAL_CUT_VERIFY="1")
            with (args.output/f"k{k}_refine.log").open("w") as log:
                run=subprocess.run([str(root/"build-gh200/single_gpu_lp_baseline"),str(ptr),str(idx),str(k),str(args.output/f"k{k}_refined.parts"),"30","50","1","1.10"],env=env,stdout=log,stderr=subprocess.STDOUT)
            result["refine_exit_code"]=run.returncode
            if run.returncode==0:
                text=(args.output/f"k{k}_refine.log").read_text()
                final=dict(re.findall(r"(\w+)=([^\s]+)",next(x for x in text.splitlines() if x.startswith("final_cut="))))
                assert final["feasible"]=="1"
                result["refined_cut"]=int(final["final_cut"])//2;result["refined_ratio"]=int(final["final_cut"])/len(original_idx)
                result["refine_seconds"]=float(re.search(r"partition_seconds=(\S+)",text).group(1))
        summary.write(json.dumps(result)+"\n");summary.flush();print(result,flush=True)
        nodes=children
