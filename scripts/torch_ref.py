#!/usr/bin/env python3
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

MODE = sys.argv[1] if len(sys.argv) > 1 else "dump"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "models/mamba.safetensors"
TOKENS = [1, 2, 3, 4]
EPS = 1e-5

W = load_file(MODEL)
d_model = W["backbone.embeddings.weight"].shape[1]
d_inner = W["backbone.layers.0.mixer.A_log"].shape[0]
d_state = W["backbone.layers.0.mixer.A_log"].shape[1]
d_conv = W["backbone.layers.0.mixer.conv1d.weight"].shape[2]
dt_rank = W["backbone.layers.0.mixer.x_proj.weight"].shape[0] - 2 * d_state
n_layers = 1 + max(int(k.split(".")[2]) for k in W if k.startswith("backbone.layers."))


def rmsnorm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def mixer(u, p):
    seq = u.shape[0]
    x, z = (u @ W[p + "in_proj.weight"].T).split(d_inner, dim=-1)
    xc = x.transpose(0, 1).unsqueeze(0)
    xc = F.conv1d(xc, W[p + "conv1d.weight"], W[p + "conv1d.bias"],
                  padding=d_conv - 1, groups=d_inner)[..., :seq]
    x = F.silu(xc.squeeze(0).transpose(0, 1))
    dt, b_mat, c_mat = (x @ W[p + "x_proj.weight"].T).split([dt_rank, d_state, d_state], dim=-1)
    dt = F.softplus(dt @ W[p + "dt_proj.weight"].T + W[p + "dt_proj.bias"])
    a_mat = -torch.exp(W[p + "A_log"])
    h = torch.zeros(d_inner, d_state)
    ys = []
    for t in range(seq):
        dA = torch.exp(dt[t].unsqueeze(1) * a_mat)
        dBu = dt[t].unsqueeze(1) * b_mat[t].unsqueeze(0) * x[t].unsqueeze(1)
        h = dA * h + dBu
        ys.append((h * c_mat[t].unsqueeze(0)).sum(1))
    y = (torch.stack(ys) + x * W[p + "D"]) * F.silu(z)
    return y @ W[p + "out_proj.weight"].T


def forward():
    u = W["backbone.embeddings.weight"][torch.tensor(TOKENS)].float()
    for i in range(n_layers):
        p = f"backbone.layers.{i}."
        u = u + mixer(rmsnorm(u, W[p + "norm.weight"]), p + "mixer.")
    return rmsnorm(u, W["backbone.norm_f.weight"])


with torch.no_grad():
    if MODE == "bench":
        torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
        for _ in range(3):
            forward()
        t0 = time.perf_counter()
        for _ in range(20):
            forward()
        ms = (time.perf_counter() - t0) / 20 * 1e3
        print(f"PyTorch forward: {ms:.3f} ms/call | {n_layers} layers "
              f"seq={len(TOKENS)} threads={torch.get_num_threads()}")
    else:
        out = sys.argv[3] if len(sys.argv) > 3 else "models/baseline"
        h = forward().numpy().astype(np.float32)
        np.save(out + ".npy", h)
        h.tofile(out + ".bin")
        print(f"baseline ||h||={np.linalg.norm(h):.6f} -> {out}.{{npy,bin}} ({h.size} floats)")
