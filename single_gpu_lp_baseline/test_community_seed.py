from array import array
from pathlib import Path
import re
import subprocess
import tempfile

binary=Path(__file__).resolve().parent/"build-gh200/community_seed"
with tempfile.TemporaryDirectory(prefix="community-lp-test-") as d:
    root=Path(d);n=32
    adj=[[u for u in range(n) if u!=v and (u//8==v//8 or abs(u-v)==1)] for v in range(n)]
    ptr=[0];idx=[]
    for row in adj:idx.extend(row);ptr.append(len(idx))
    for name,code,values in [("ptr","q",ptr),("idx","q",idx),("initial","i",[v%2 for v in range(n)])]:
        with (root/name).open("wb") as f:array(code,values).tofile(f)
    for seed in range(5):
        run=subprocess.run([str(binary),str(root/"ptr"),str(root/"idx"),str(root/"initial"),str(root/"out"),"2","8","5","10",str(seed)],capture_output=True,text=True,check=True)
        out=array("i");out.frombytes((root/"out").read_bytes());assert len(out)==n and set(out)=={0,1}
        assert max(out.count(0),out.count(1))<=17
        cut=sum(out[v]!=out[u] for v in range(n) for u in adj[v])//2
        assert cut==int(re.search(r"final_cut=(\d+)",run.stdout).group(1))
    print("PASS: five planted-graph seeds, full cut and capacity verified")
