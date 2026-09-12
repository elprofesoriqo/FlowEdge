"""Deterministic PyTorch Mamba checkpoint reference."""

import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class MambaReference:
    tokens = (1, 2, 3, 4)
    eps = 1e-5

    def __init__(self, model):
        self.weights = {key: value.float() for key, value in load_file(model).items()}
        weights = self.weights
        self.d_inner, self.d_state = weights["backbone.layers.0.mixer.A_log"].shape
        self.d_conv = weights["backbone.layers.0.mixer.conv1d.weight"].shape[2]
        self.dt_rank = (
            weights["backbone.layers.0.mixer.x_proj.weight"].shape[0] - 2 * self.d_state
        )
        self.n_layers = 1 + max(
            int(key.split(".")[2])
            for key in weights
            if key.startswith("backbone.layers.")
        )

    def rmsnorm(self, x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * weight

    def mixer(self, u, prefix):
        weights, seq = self.weights, u.shape[0]
        x, z = (u @ weights[prefix + "in_proj.weight"].T).split(self.d_inner, dim=-1)
        xc = F.conv1d(
            x.T.unsqueeze(0),
            weights[prefix + "conv1d.weight"],
            weights[prefix + "conv1d.bias"],
            padding=self.d_conv - 1,
            groups=self.d_inner,
        )[..., :seq]
        x = F.silu(xc.squeeze(0).T)
        dt, b_mat, c_mat = (x @ weights[prefix + "x_proj.weight"].T).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(
            dt @ weights[prefix + "dt_proj.weight"].T + weights[prefix + "dt_proj.bias"]
        )
        a_mat, h, ys = (
            -torch.exp(weights[prefix + "A_log"]),
            torch.zeros(self.d_inner, self.d_state),
            [],
        )
        for t in range(seq):
            h = torch.exp(dt[t].unsqueeze(1) * a_mat) * h + dt[t].unsqueeze(1) * b_mat[
                t
            ].unsqueeze(0) * x[t].unsqueeze(1)
            ys.append((h * c_mat[t].unsqueeze(0)).sum(1))
        return (
            (torch.stack(ys) + x * weights[prefix + "D"])
            * F.silu(z)
            @ weights[prefix + "out_proj.weight"].T
        )

    def forward(self):
        weights = self.weights
        u = weights["backbone.embeddings.weight"][torch.tensor(self.tokens)].float()
        for index in range(self.n_layers):
            prefix = f"backbone.layers.{index}."
            u += self.mixer(
                self.rmsnorm(u, weights[prefix + "norm.weight"]), prefix + "mixer."
            )
        return self.rmsnorm(u, weights["backbone.norm_f.weight"])

    def bench(self, threads):
        torch.set_num_threads(threads)
        with torch.no_grad():
            for _ in range(3):
                self.forward()
            start = time.perf_counter()
            for _ in range(20):
                self.forward()
        ms = (time.perf_counter() - start) / 20 * 1e3
        print(
            f"PyTorch forward: {ms:.3f} ms/call | {self.n_layers} layers "
            f"seq={len(self.tokens)} threads={torch.get_num_threads()}"
        )

    def dump(self, output):
        with torch.no_grad():
            h = self.forward().numpy().astype(np.float32)
        np.save(output + ".npy", h)
        h.tofile(output + ".bin")
        print(
            f"baseline ||h||={np.linalg.norm(h):.6f} -> "
            f"{output}.{{npy,bin}} ({h.size} floats)"
        )
        if "flow.in_proj.weight" in self.weights:
            self.dump_action(h[-1], output)

    def dump_action(self, condition, output):
        weights = self.weights
        action_size, time_size = (
            weights["flow.in_proj.weight"].shape[1],
            weights["flow.time_proj.weight"].shape[1],
        )
        freqs = torch.tensor(
            [10000.0 ** (-index / (time_size // 2)) for index in range(time_size // 2)]
        )
        x = torch.cos(torch.arange(action_size, dtype=torch.float32) * 0.3) * 0.5
        for step in range(10):
            x += 0.1 * self.velocity(x, step / 10.0, condition, freqs)
        action = x.numpy().astype(np.float32)
        action.tofile(output + "_action.bin")
        values = [round(value, 6) for value in action[:4].tolist()]
        print(f"baseline action[:4] = {values}")

    def velocity(self, x, time_value, condition, freqs):
        weights = self.weights
        hidden = F.silu(
            weights["flow.in_proj.weight"] @ x
            + weights["flow.cond_proj.weight"] @ condition
            + weights["flow.time_proj.weight"]
            @ torch.cat([torch.sin(time_value * freqs), torch.cos(time_value * freqs)])
        )
        for index in range(sum(key.startswith("flow.layers.") for key in weights)):
            hidden = F.silu(weights[f"flow.layers.{index}.weight"] @ hidden)
        return weights["flow.out_proj.weight"] @ hidden
