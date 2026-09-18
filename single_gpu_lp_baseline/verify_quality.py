"""Independent full-CSR cut and upper-capacity validation of int32 labels."""
import json
import math
from pathlib import Path
import sys
import numpy as np

data=Path(sys.argv[1]); dataset=data.name;k=int(sys.argv[2])
ptr=np.memmap(data/f"{dataset}_sym_indptr.bin",dtype="<i8",mode="r")
idx=np.memmap(data/f"{dataset}_sym_indices.bin",dtype="<i8",mode="r")
n=len(ptr)-1
assert ptr[0]==0 and ptr[-1]==len(idx)
for name in sys.argv[3:]:
    labels=np.fromfile(name,dtype="<i4")
    assert len(labels)==n and np.all((labels>=0)&(labels<k))
    loads=np.bincount(labels,minlength=k)
    assert int(loads.max())<=math.floor(n/k*1.1)
    cut=0
    for first in range(0,n,8192):
        last=min(n,first+8192)
        sources=np.repeat(labels[first:last],np.diff(ptr[first:last+1]))
        cut+=int(np.count_nonzero(sources!=labels[idx[ptr[first]:ptr[last]]]))
    assert cut%2==0
    print(json.dumps(dict(parts=name,edge_cut=cut//2,cut_ratio=cut/len(idx),loads=loads.tolist(),vertex_imbalance=float(loads.max())*k/n,feasible=True)),flush=True)
