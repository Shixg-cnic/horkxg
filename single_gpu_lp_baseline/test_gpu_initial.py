"""Validate external-label loading and final feasibility enforcement on GPU."""
from array import array
import os
from pathlib import Path
import subprocess
import tempfile

binary=Path(__file__).resolve().parent/"build-gh200/single_gpu_lp_baseline"
with tempfile.TemporaryDirectory(prefix="lp-initial-test-") as directory:
    d=Path(directory)
    for name,code,values in [("ptr","q",list(range(0,17,2))),("idx","q",[u for v in range(8) for u in [(v-1)%8,(v+1)%8]])]:
        with (d/name).open("wb") as f:array(code,values).tofile(f)
    env=os.environ.copy();env.update(INITIAL_PARTITION=str(d/"initial"),GLOBAL_CYCLES="0",POLISH_ROUNDS="0")
    for labels,success in [([0]*4+[1]*4,True),([0]*6+[1]*2,False),([0]*7+[2],False)]:
        with (d/"initial").open("wb") as f:array("i",labels).tofile(f)
        run=subprocess.run([str(binary),str(d/"ptr"),str(d/"idx"),"2",str(d/"out"),"30","0","1","1.10"],env=env,capture_output=True,text=True)
        assert (run.returncode==0)==success,(run.stdout,run.stderr)
        if success:
            assert (d/"initial").read_bytes()==(d/"out").read_bytes()
            assert "final_cut=4 " in run.stdout and "distance_seed index=" not in run.stdout
    print("PASS: external initialization, exact passthrough cut, capacity and label rejection")
