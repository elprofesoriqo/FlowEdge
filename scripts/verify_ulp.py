#!/usr/bin/env python3

import sys
import os

if os.path.exists("build") and "build" not in sys.path:
    sys.path.append("build")
if os.path.exists("build/Release") and "build/Release" not in sys.path:
    sys.path.append("build/Release")
if os.name == "nt":
    from pathlib import Path

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        winpthread = Path(entry) / "libwinpthread-1.dll"
        if winpthread.is_file():
            os.add_dll_directory(str(winpthread.parent))
            break

import numpy as np
import flowedge  # built with -DFLOWEDGE_PYTHON=ON

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/mamba_flow.safetensors"
PREFIX = np.array([1, 2, 3, 4], dtype=np.int32)  # must match scripts/torch_ref.py TOKENS
NFE = 10
EPS_REL = 2e-3  # per-dim relative-error gate
EPS_ULP = 4096  # per-dim float32 ULP gate

import torch_ref
sys.argv = ["torch_ref.py", "dump", MODEL, "build/ulp_ref"]  # main() reads all config from argv
torch_ref.main()
torch_action = np.fromfile("build/ulp_ref_action.bin", dtype=np.float32)

eng = flowedge.Engine(MODEL)
a = eng.action_dim
if a == 0:
    print("checkpoint has no flow head")
    sys.exit(1)
noise = (np.cos(np.arange(a, dtype=np.float32) * 0.3) * 0.5).astype(np.float32)
cpp_action = np.asarray(eng.sample(PREFIX, noise, NFE, "euler"), dtype=np.float32)

rel = np.abs(cpp_action - torch_action) / (np.abs(torch_action) + 1e-8)
ulp = np.abs(cpp_action - torch_action) / np.spacing(np.maximum(np.abs(cpp_action), np.abs(torch_action)))
print(f"action_dim={a}  max rel-error={rel.max():.2e} (<{EPS_REL})  max ULP={int(ulp.max())} (<{EPS_ULP})")
ok = bool(rel.max() < EPS_REL and ulp.max() < EPS_ULP)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
