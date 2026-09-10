#!/usr/bin/env python3
"""
Usage: python flow_sample.py <mamba_flow.safetensors> [euler|heun|rk4] [steps]
"""
import os
from pathlib import Path
import sys
import numpy as np

if os.name == "nt":
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        winpthread = Path(entry) / "libwinpthread-1.dll"
        if winpthread.is_file():
            os.add_dll_directory(str(winpthread.parent))
            break

try:
    import flowedge
except ImportError:
    print("Error: 'flowedge' python package is not installed.")
    print("Run 'pip install .' from the repository root to build and install it.")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python flow_sample.py <model.safetensors> [euler|heun|rk4] [steps]")
        sys.exit(1)
        
    model_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "euler"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    if method not in {"euler", "heun", "rk4"}:
        print("Error: solver must be euler, heun, or rk4")
        sys.exit(1)
    
    try:
        e = flowedge.Engine(model_path)
    except Exception as ex:
        print(f"Error loading model: {ex}")
        sys.exit(1)
        
    a = e.action_dim
    if a == 0:
        print("error: checkpoint has no flow head")
        sys.exit(1)
        
    prefix = np.array([1, 2, 3, 4], dtype=np.int32)
    noise = np.random.randn(a).astype(np.float32)
    
    try:
        action = e.sample(prefix=prefix, noise=noise, steps=steps, method=method)
        print(f"action_dim={a}  solver={method}  NFE={steps}  action[0..2]={action[0]:.4f}, {action[1 % a]:.4f}, {action[2 % a]:.4f}")
    except Exception as ex:
        print(f"sample failed: {ex}")
        sys.exit(1)

if __name__ == "__main__":
    main()
