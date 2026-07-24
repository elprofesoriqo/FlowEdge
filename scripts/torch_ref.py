#!/usr/bin/env python3
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

def main():
    global MODE, MODEL, TOKENS, EPS, W, Wf, lyr, cond, x0, freqs
    MODE = sys.argv[1] if len(sys.argv) > 1 else "dump"
    MODEL = sys.argv[2] if len(sys.argv) > 2 else "models/mamba.safetensors"
    TOKENS = [1, 2, 3, 4]
    EPS = 1e-5
    
    if MODE == "latency":  # flow head + Euler ODE per-call latency distribution
        torch.set_num_threads(1)
        A, C, H, T, L, N = 32, 768, 256, 128, 4, 10
        torch.manual_seed(0)
        Wf = {"in": torch.randn(H, A) * 0.02, "time": torch.randn(H, T) * 0.02,
              "cond": torch.randn(H, C) * 0.02, "out": torch.randn(A, H) * 0.02}
        lyr = [torch.randn(H, H) * 0.02 for _ in range(L)]
        cond, x0 = torch.full((C,), 0.1), torch.full((A,), 0.1)
        freqs = torch.tensor([10000.0 ** (-i / (T // 2)) for i in range(T // 2)])
    
        def vel(x, t):
            h = Wf["in"] @ x + Wf["cond"] @ cond \
                + Wf["time"] @ torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)])
            h = F.silu(h)
            for w in lyr:
                h = F.silu(w @ h)
            return Wf["out"] @ h
    
        def sample():
            x = x0.clone()
            for k in range(N):
                x = x + (1.0 / N) * vel(x, k / N)
            return x
    
        with torch.no_grad():
            for _ in range(50):
                sample()
            iters = 5000
            lat = np.empty(iters)
            for it in range(iters):
                t0 = time.perf_counter()
                sample()
                lat[it] = (time.perf_counter() - t0) * 1e6
        lat.sort()
        p50, p99, p999 = (float(np.percentile(lat, q)) for q in (50, 99, 99.9))
        print(f"PyTorch    | {lat.mean():9.2f} | {p50:9.2f} | {p99:9.2f} | {p999:9.2f} | {lat[0]:9.2f} | {lat[-1]:9.2f} | >0")
        sys.exit(0)
    
    W = load_file(MODEL)
    W = {k: v.float() for k, v in W.items()}
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
            ht = forward()  # [seq, d_model], post norm_f
            h = ht.numpy().astype(np.float32)
            np.save(out + ".npy", h)
            h.tofile(out + ".bin")
            print(f"baseline ||h||={np.linalg.norm(h):.6f} -> {out}.{{npy,bin}} ({h.size} floats)")
    
            if "flow.in_proj.weight" in W:
                Af = W["flow.in_proj.weight"].shape[1]
                Tf = W["flow.time_proj.weight"].shape[1]
                Lf = sum(1 for k in W if k.startswith("flow.layers."))
                cond = ht[-1]  # last-token hidden = conditioning vector
                freqs = torch.tensor([10000.0 ** (-i / (Tf // 2)) for i in range(Tf // 2)])
    
                def fvel(x, t):
                    hh = W["flow.in_proj.weight"] @ x + W["flow.cond_proj.weight"] @ cond \
                        + W["flow.time_proj.weight"] @ torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)])
                    hh = F.silu(hh)
                    for lf in range(Lf):
                        hh = F.silu(W[f"flow.layers.{lf}.weight"] @ hh)
                    return W["flow.out_proj.weight"] @ hh
    
                x = torch.cos(torch.arange(Af, dtype=torch.float32) * 0.3) * 0.5  # deterministic noise
                for k in range(10):
                    x = x + 0.1 * fvel(x, k / 10.0)  # Euler, N=10
                action = x.numpy().astype(np.float32)
                action.tofile(out + "_action.bin")
                print(f"baseline action[:4] = {[round(v, 6) for v in action[:4].tolist()]}")

if __name__ == '__main__':
    main()
