#!/usr/bin/env python3
"""Parity-check the pinned public LeRobot diffusion_pusht U-Net checkpoint.

The RGB encoder is intentionally ignored: FlowEdge consumes the already
flattened observation condition. The script is opt-in because loading the
public FP32 checkpoint requires about 2 GiB of working memory.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_diffusion import ConditionalUnet1d, reference_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="original LeRobot model.safetensors")
    parser.add_argument("converted", type=Path, help="FlowEdge converted checkpoint")
    parser.add_argument("--build", type=Path, required=True, help="directory containing flowedge")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.build.resolve()))
    import flowedge

    config_path = args.source.with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    down_dims = [int(value) for value in config["down_dims"]]
    action_dim = int(config["output_features"]["action"]["shape"][0])
    horizon = int(config["horizon"])
    timestep_dim = int(config["diffusion_step_embed_dim"])
    groups = int(config["n_groups"])
    condition_dim = 132  # pinned public model: two observations x (state + image features)
    torch.set_num_threads(max(1, args.threads))

    tensors = load_file(str(args.source), device="cpu")
    unet_state = {
        key.removeprefix("diffusion.unet."): value
        for key, value in tensors.items()
        if key.startswith("diffusion.unet.")
    }
    action_min = tensors["unnormalize_outputs.buffer_action.min"].float()
    action_max = tensors["unnormalize_outputs.buffer_action.max"].float()
    model = ConditionalUnet1d(
        action_dim, condition_dim, down_dims, int(config["kernel_size"]), groups, timestep_dim
    ).eval()
    model.load_state_dict(unet_state, strict=True)
    del tensors, unet_state

    engine = flowedge.Engine(str(args.converted), threads=args.threads)
    metadata = engine.diffusion_metadata
    assert metadata["horizon"] == horizon and metadata["action_dim"] == action_dim
    assert metadata["condition_dim"] == condition_dim
    condition = np.linspace(-0.25, 0.45, condition_dim, dtype=np.float32).reshape(1, -1)
    noise = np.sin(np.arange(horizon * action_dim, dtype=np.float32) * 0.17).reshape(
        1, horizon, action_dim
    )
    condition_tensor = torch.from_numpy(condition)
    noise_tensor = torch.from_numpy(noise)
    for timestep in (99.0, 50.0, 0.0):
        with torch.no_grad():
            expected = model(noise_tensor, torch.tensor([timestep]), condition_tensor).numpy()[0]
        actual = engine.diffusion_denoise(condition, noise, timestep)
        error = float(np.max(np.abs(actual - expected)))
        assert np.allclose(actual, expected, atol=3e-4, rtol=3e-4), (timestep, error)
        print(f"public denoiser t={timestep:g} max_abs={error:.3g}")

    for steps in (1, 10):
        with torch.no_grad():
            expected = reference_sample(
                model,
                condition_tensor,
                noise_tensor,
                steps,
                "ddim",
                0,
                config,
                action_min,
                action_max,
            )
        actual = engine.sample_diffusion(condition, noise, steps, "ddim", 0)
        error = float(np.max(np.abs(actual - expected)))
        print(f"public DDIM steps={steps} actual={actual[0].tolist()} expected={expected[0].tolist()}")
        assert np.allclose(actual, expected, atol=6e-4, rtol=6e-4), (steps, error)
        assert np.array_equal(actual, engine.sample_diffusion(condition, noise, steps, "ddim", 0))
        print(f"public DDIM steps={steps} max_abs={error:.3g}")


if __name__ == "__main__":
    main()
