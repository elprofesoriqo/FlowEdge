#!/usr/bin/env python3
import os
import sys

import torch
from safetensors.torch import save_file

OUT = sys.argv[1] if len(sys.argv) > 1 else "models/synthetic.safetensors"
V, DM, DI, DS, DC, DR, NL = 100, 64, 128, 8, 4, 8, 2
torch.manual_seed(0)


def r(*shape):
    return (torch.randn(*shape) * 0.1).float()


W = {
    "backbone.embeddings.weight": r(V, DM),
    "backbone.norm_f.weight": torch.ones(DM),
}
for i in range(NL):
    p = f"backbone.layers.{i}."
    W[p + "norm.weight"] = torch.ones(DM)
    W[p + "mixer.in_proj.weight"] = r(2 * DI, DM)
    W[p + "mixer.conv1d.weight"] = r(DI, 1, DC)
    W[p + "mixer.conv1d.bias"] = r(DI)
    W[p + "mixer.x_proj.weight"] = r(DR + 2 * DS, DI)
    W[p + "mixer.dt_proj.weight"] = r(DI, DR)
    W[p + "mixer.dt_proj.bias"] = r(DI)
    W[p + "mixer.A_log"] = (torch.randn(DI, DS) * 0.5).float()
    W[p + "mixer.D"] = r(DI)
    W[p + "mixer.out_proj.weight"] = r(DM, DI)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
save_file(W, OUT)
print(f"wrote {OUT}: vocab={V} d_model={DM} d_inner={DI} d_state={DS} "
      f"d_conv={DC} dt_rank={DR} n_layers={NL}")