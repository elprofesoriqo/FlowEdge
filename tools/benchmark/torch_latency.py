#!/usr/bin/env python3
"""Deterministic PyTorch flow-head latency reference."""

import time

import numpy as np
import torch
import torch.nn.functional as F


def sample(velocity, initial, method, steps):
    x = initial.clone()
    for index in range(steps):
        time_value, dt = index / steps, 1.0 / steps
        k1 = velocity(x, time_value)
        if method == "euler":
            x += dt * k1
        elif method == "heun":
            x += 0.5 * dt * (k1 + velocity(x + dt * k1, time_value + dt))
        else:
            half = 0.5 * dt
            k2 = velocity(x + half * k1, time_value + half)
            k3 = velocity(x + half * k2, time_value + half)
            x += (
                dt
                / 6.0
                * (k1 + 2.0 * k2 + 2.0 * k3 + velocity(x + dt * k3, time_value + dt))
            )
    return x


def run(args):
    torch.set_num_threads(1)
    A, C, H, T, L, N = 32, 768, 256, 128, 4, 10
    method = args[1] if len(args) > 1 else "euler"
    iters = int(args[2]) if len(args) > 2 else 5000
    if method not in {"euler", "heun", "rk4"}:
        raise SystemExit("latency method must be euler, heun, or rk4")
    if iters <= 0:
        raise SystemExit("latency iterations must be positive")

    # Match bench/latency_bench.cc exactly: same tensor shapes, deterministic
    # row-major values, input vectors, and 10 ODE steps. The two runtimes can
    # then be compared without a random-weight or workload-shape confounder.
    def filled(*shape):
        n = int(np.prod(shape))
        values = 0.02 * (torch.arange(n, dtype=torch.float32) % 17 - 8)
        return values.reshape(shape)

    Wf = {
        "in": filled(H, A),
        "time": filled(H, T),
        "cond": filled(H, C),
        "out": filled(A, H),
    }
    lyr = [filled(H, H) for _ in range(L)]
    cond, x0 = torch.full((C,), 0.1), torch.full((A,), 0.1)
    c_emb = Wf["cond"] @ cond  # FlowEdge computes this once per sample.
    freqs = torch.tensor([10000.0 ** (-i / (T // 2)) for i in range(T // 2)])

    def vel(x, t):
        h = (
            Wf["in"] @ x
            + c_emb
            + Wf["time"] @ torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)])
        )
        h = F.silu(h)
        for w in lyr:
            h = F.silu(w @ h)
        return Wf["out"] @ h

    with torch.no_grad():
        for _ in range(50):
            sample(vel, x0, method, N)
        lat = np.empty(iters)
        for it in range(iters):
            t0 = time.perf_counter()
            sample(vel, x0, method, N)
            lat[it] = (time.perf_counter() - t0) * 1e6
    lat.sort()
    p50, p99, p999 = (lat[int(q * (iters - 1))] for q in (0.50, 0.99, 0.999))
    print(
        f"PyTorch    | {lat.mean():9.2f} | {p50:9.2f} | {p99:9.2f} | "
        f"{p999:9.2f} | {lat[0]:9.2f} | {lat[-1]:9.2f} | >0"
    )
    return
