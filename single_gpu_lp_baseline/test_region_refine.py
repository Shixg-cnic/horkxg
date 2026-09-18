from array import array
from pathlib import Path
import random
import re
import subprocess
import tempfile

binary=Path(__file__).resolve().parent/"build-gh200/region_refine"
with tempfile.TemporaryDirectory(prefix="region-cut-test-") as directory:
    d=Path(directory)
    for seed in range(20):
        rng=random.Random(seed);n=64;k=[2,4,8][seed%3];adj=[[] for _ in range(n)]
        for v in range(n):
            for u in range(v+1,n):
                if rng.random()<(0.65 if v//8==u//8 else 0.04):adj[v].append(u);adj[u].append(v)
        labels=[v%k for v in range(n)];rng.shuffle(labels);ptr=[0];idx=[]
        for row in adj:idx.extend(row);ptr.append(len(idx))
        for name,code,values in [("ptr","q",ptr),("idx","q",idx),("in","i",labels)]:
            with (d/name).open("wb") as f:array(code,values).tofile(f)
        for size in [8,32,64]:
            run=subprocess.run([str(binary),str(d/"ptr"),str(d/"idx"),str(d/"in"),str(d/"out"),str(k),str(size),"3",str(seed)],capture_output=True,text=True,check=True)
            result=array("i");result.frombytes((d/"out").read_bytes())
            def cut(part):return sum(part[v]!=part[u] for v in range(n) for u in adj[v])//2
            assert len(result)==n and all(0<=p<k for p in result)
            assert all(result.count(p)<=int(n/k*1.1) for p in range(k))
            assert cut(result)<=cut(labels)
            assert cut(result)==int(re.search(r"final_cut=(\d+)",run.stdout).group(1))
    print("PASS: 60 regional-search runs; exact cuts, capacity, non-regression")
