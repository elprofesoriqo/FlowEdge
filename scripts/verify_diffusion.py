#!/usr/bin/env python3
"""Tiny end-to-end ConditionalUnet1D parity check for converter + Python API."""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timestep):
        half = self.dim // 2
        factor = math.log(10000) / (half - 1)
        frequencies = torch.exp(torch.arange(half, device=timestep.device) * -factor)
        embedding = timestep[:, None] * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, groups):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim, kernel, groups):
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel, groups)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel, groups)
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(condition_dim, 2 * out_channels))
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, condition):
        out = self.conv1(x)
        scale, bias = self.cond_encoder(condition).unsqueeze(-1).chunk(2, dim=1)
        out = scale * out + bias
        return self.conv2(out) + self.residual_conv(x)


class ConditionalUnet1d(nn.Module):
    def __init__(self, action_dim, condition_dim, down_dims, kernel, groups, timestep_dim):
        super().__init__()
        condition_features = timestep_dim + condition_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(timestep_dim),
            nn.Linear(timestep_dim, timestep_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_dim * 4, timestep_dim),
        )
        in_out = [(action_dim, down_dims[0]), *zip(down_dims[:-1], down_dims[1:])]
        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in, dim_out, condition_features, kernel, groups),
                        ConditionalResidualBlock1d(dim_out, dim_out, condition_features, kernel, groups),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1)
                        if index + 1 < len(in_out)
                        else nn.Identity(),
                    ]
                )
            )
        largest = down_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(largest, largest, condition_features, kernel, groups),
                ConditionalResidualBlock1d(largest, largest, condition_features, kernel, groups),
            ]
        )
        self.up_modules = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(2 * dim_out, dim_in, condition_features, kernel, groups),
                        ConditionalResidualBlock1d(dim_in, dim_in, condition_features, kernel, groups),
                        nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel, 8),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, sample, timestep, condition):
        x = sample.moveaxis(-1, -2)
        time_embedding = self.diffusion_step_encoder(timestep)
        features = torch.cat((time_embedding, condition), dim=-1)
        skips = []
        for first, second, downsample in self.down_modules:
            x = first(x, features)
            x = second(x, features)
            skips.append(x)
            x = downsample(x)
        for block in self.mid_modules:
            x = block(x, features)
        for first, second, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = first(x, features)
            x = second(x, features)
            x = upsample(x)
        return self.final_conv(x).moveaxis(-1, -2)


def alpha_schedule(train_steps):
    def alpha_bar(time):
        return math.cos((time + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = torch.tensor(
        [
            min(1 - alpha_bar((step + 1) / train_steps) / alpha_bar(step / train_steps), 0.999)
            for step in range(train_steps)
        ],
        dtype=torch.float32,
    )
    return torch.cumprod(1 - betas, dim=0)


class GaussianGenerator:
    def __init__(self, seed):
        self.state = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        self.spare = np.float32(0)
        self.has_spare = False

    def integer(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF

    def uniform(self):
        return np.float32(((self.integer() >> 40) + 1) / 16777217.0)

    def next(self):
        if self.has_spare:
            self.has_spare = False
            return self.spare
        first = max(self.uniform(), np.finfo(np.float32).tiny)
        second = self.uniform()
        radius = np.sqrt(np.float32(-2) * np.log(first), dtype=np.float32)
        angle = np.float32(2 * math.pi) * second
        self.spare = radius * np.sin(angle, dtype=np.float32)
        self.has_spare = True
        return radius * np.cos(angle, dtype=np.float32)


@torch.no_grad()
def reference_sample(model, condition, noise, inference_steps, scheduler, seed, config, action_min, action_max):
    alphas = alpha_schedule(config["num_train_timesteps"])
    sample = noise.clone()
    ratio = config["num_train_timesteps"] // inference_steps
    gaussian = GaussianGenerator(seed)
    for reverse_index in reversed(range(inference_steps)):
        timestep = reverse_index * ratio
        previous = timestep - ratio
        predicted_noise = model(
            sample,
            torch.tensor([float(timestep)], dtype=torch.float32),
            condition,
        )
        alpha_t = alphas[timestep]
        alpha_previous = alphas[previous] if previous >= 0 else torch.tensor(1.0)
        beta_t = 1 - alpha_t
        predicted_original = (sample - beta_t.sqrt() * predicted_noise) / alpha_t.sqrt()
        predicted_original = predicted_original.clamp(
            -config["clip_sample_range"], config["clip_sample_range"]
        )
        if scheduler == "ddim":
            sample = alpha_previous.sqrt() * predicted_original + (1 - alpha_previous).sqrt() * predicted_noise
        else:
            current_alpha = alpha_t / alpha_previous
            current_beta = 1 - current_alpha
            original_coefficient = alpha_previous.sqrt() * current_beta / beta_t
            sample_coefficient = current_alpha.sqrt() * (1 - alpha_previous) / beta_t
            sample = original_coefficient * predicted_original + sample_coefficient * sample
            if previous >= 0:
                variance = ((1 - alpha_previous) / beta_t) * current_beta
                step_noise = torch.tensor(
                    [gaussian.next() for _ in range(sample.numel())], dtype=torch.float32
                ).reshape_as(sample)
                sample = sample + variance.clamp_min(1e-20).sqrt() * step_noise
    return ((sample + 1) * 0.5 * (action_max - action_min) + action_min).numpy()[0]


def _write_modern_processor_fixture(directory, state, config, action_min, action_max):
    modern_directory = directory / "modern-processor"
    modern_directory.mkdir()
    modern_state = {
        key: value for key, value in state.items() if not key.startswith("unnormalize_outputs.")
    }
    save_file(modern_state, modern_directory / "model.safetensors")
    modern_config = dict(config)
    modern_config.pop("normalization_mapping")
    (modern_directory / "config.json").write_text(
        json.dumps(modern_config), encoding="utf-8"
    )
    (modern_directory / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_postprocessor",
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "config": {
                            "eps": 1e-8,
                            "features": {"action": {"type": "ACTION", "shape": [2]}},
                            "norm_map": {
                                "VISUAL": "MEAN_STD",
                                "STATE": "MIN_MAX",
                                "ACTION": "MIN_MAX",
                            },
                        },
                        "state_file": "policy_postprocessor_step_0.safetensors",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {"action.min": action_min, "action.max": action_max},
        modern_directory / "policy_postprocessor_step_0.safetensors",
    )
    return modern_directory


def main():
    root = Path(__file__).resolve().parents[1]
    build_dir = None
    if len(sys.argv) > 1:
        build_dir = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(build_dir))
    import flowedge

    config = {
        "type": "diffusion",
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "epsilon",
        "use_film_scale_modulation": True,
        "use_group_norm": True,
        "normalization_mapping": {"ACTION": "MIN_MAX"},
        "down_dims": [8, 16],
        "output_features": {"action": {"shape": [2], "type": "ACTION"}},
        "horizon": 4,
        "n_action_steps": 2,
        "n_obs_steps": 2,
        "kernel_size": 3,
        "n_groups": 2,
        "diffusion_step_embed_dim": 4,
        "num_train_timesteps": 10,
        "clip_sample": True,
        "clip_sample_range": 1.0,
    }
    torch.manual_seed(17)
    model = ConditionalUnet1d(2, 3, [8, 16], 3, 2, 4).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(0.2)
    state = {f"diffusion.unet.{name}": value for name, value in model.state_dict().items()}
    action_min = torch.tensor([-2.0, 0.0], dtype=torch.float32)
    action_max = torch.tensor([2.0, 10.0], dtype=torch.float32)
    state["unnormalize_outputs.buffer_action.min"] = action_min
    state["unnormalize_outputs.buffer_action.max"] = action_max

    with tempfile.TemporaryDirectory(prefix="flowedge-diffusion-") as directory:
        directory = Path(directory)
        source = directory / "model.safetensors"
        converted = directory / "flowedge.safetensors"
        save_file(state, source)
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        subprocess.run(
            [sys.executable, str(root / "convert" / "convert.py"), str(directory), str(converted),
             "--arch", "diffusion"],
            check=True,
        )
        if build_dir is not None:
            inspector_candidates = [
                build_dir / "flowedge-inspect",
                build_dir / "flowedge-inspect.exe",
                build_dir / "Release" / "flowedge-inspect.exe",
            ]
            inspector = next((path for path in inspector_candidates if path.is_file()), None)
            if inspector is not None:
                inspection = subprocess.run(
                    [str(inspector), str(converted), "--json"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                report = json.loads(inspection.stdout)
                model = report["model"]
                assert model["family"] == "diffusion-policy"
                assert model["action_dim"] == 2
                assert model["action_horizon"] == 4
                assert model["diffusion_stages"] == 2
                assert report["compatibility"]["supported"] is True
                print("diffusion checkpoint preflight OK")
        modern_directory = _write_modern_processor_fixture(
            directory, state, config, action_min, action_max
        )
        modern_converted = directory / "modern-flowedge.safetensors"
        subprocess.run(
            [
                sys.executable,
                str(root / "convert" / "convert.py"),
                str(modern_directory),
                str(modern_converted),
                "--arch",
                "diffusion",
            ],
            check=True,
        )
        legacy_output = load_file(converted)
        modern_output = load_file(modern_converted)
        assert torch.equal(legacy_output["dp.action_min"], modern_output["dp.action_min"])
        assert torch.equal(legacy_output["dp.action_max"], modern_output["dp.action_max"])
        print("modern LeRobot processor normalization conversion OK")
        unsupported_processor = json.loads(
            (modern_directory / "policy_postprocessor.json").read_text(encoding="utf-8")
        )
        unsupported_processor["steps"][0]["config"]["norm_map"]["ACTION"] = "MEAN_STD"
        unsupported_processor_path = modern_directory / "unsupported-processor.json"
        unsupported_processor_path.write_text(
            json.dumps(unsupported_processor), encoding="utf-8"
        )
        rejected_processor = subprocess.run(
            [
                sys.executable,
                str(root / "convert" / "convert.py"),
                str(modern_directory),
                str(directory / "rejected-processor.safetensors"),
                "--arch",
                "diffusion",
                "--processor",
                str(unsupported_processor_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected_processor.returncode != 0
        assert "processor ACTION normalization" in rejected_processor.stderr
        assert not (directory / "rejected-processor.safetensors").exists()
        unsupported = dict(config)
        unsupported["prediction_type"] = "sample"
        unsupported_path = directory / "unsupported.json"
        unsupported_path.write_text(json.dumps(unsupported), encoding="utf-8")
        rejected = subprocess.run(
            [
                sys.executable,
                str(root / "convert" / "convert.py"),
                str(source),
                str(directory / "rejected.safetensors"),
                "--arch",
                "diffusion",
                "--config",
                str(unsupported_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0 and "prediction_type" in rejected.stderr
        invalid_kernel = dict(config)
        invalid_kernel["kernel_size"] = 4
        invalid_kernel_path = directory / "invalid-kernel.json"
        invalid_kernel_path.write_text(json.dumps(invalid_kernel), encoding="utf-8")
        invalid = subprocess.run(
            [
                sys.executable,
                str(root / "convert" / "convert.py"),
                str(source),
                str(directory / "invalid.safetensors"),
                "--arch",
                "diffusion",
                "--config",
                str(invalid_kernel_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert invalid.returncode != 0 and "kernel_size" in invalid.stderr
        engine = flowedge.Engine(str(converted), threads=0)
        assert engine.model_metadata["architecture"] == 4
        assert engine.action_horizon == 4 and engine.action_dim == 2 and engine.condition_dim == 3
        assert engine.diffusion_metadata == {
            "protocol_version": 1,
            "action_dim": 2,
            "condition_dim": 3,
            "horizon": 4,
            "action_steps": 2,
            "observation_steps": 2,
            "train_timesteps": 10,
            "clip_sample": True,
            "clip_sample_range": 1.0,
        }
        condition = np.ascontiguousarray([[0.2, -0.4, 0.7]], dtype=np.float32)
        noise = np.ascontiguousarray(np.linspace(-0.8, 0.9, 8, dtype=np.float32).reshape(1, 4, 2))
        condition_tensor = torch.from_numpy(condition)
        noise_tensor = torch.from_numpy(noise)
        with torch.no_grad():
            expected_noise = model(noise_tensor, torch.tensor([7.0]), condition_tensor).numpy()[0]
        actual_noise = engine.diffusion_denoise(condition, noise, 7.0)
        denoiser_error = float(np.max(np.abs(actual_noise - expected_noise)))
        assert np.allclose(actual_noise, expected_noise, atol=2e-4, rtol=2e-4), (
            denoiser_error,
            actual_noise,
            expected_noise,
        )
        print(f"diffusion denoiser parity OK: max_abs={denoiser_error:.3g}")
        for scheduler, seed, tolerance in (("ddim", 1, 4e-4), ("ddpm", 42, 8e-4)):
            expected = reference_sample(
                model, condition_tensor, noise_tensor, 5, scheduler, seed, config, action_min, action_max
            )
            actual = engine.sample_diffusion(condition, noise, 5, scheduler, seed)
            error = float(np.max(np.abs(actual - expected)))
            assert np.allclose(actual, expected, atol=tolerance, rtol=tolerance), (
                scheduler,
                error,
                actual,
                expected,
            )
            repeated = engine.sample_diffusion(condition, noise, 5, scheduler, seed)
            assert np.array_equal(actual, repeated), f"{scheduler} is not deterministic"
            into = np.empty_like(actual)
            engine.sample_diffusion_into(condition, noise, into, 5, scheduler, seed)
            assert np.array_equal(actual, into), f"{scheduler} into API diverged"
            print(f"diffusion {scheduler} parity OK: max_abs={error:.3g}")
        if build_dir is not None:
            candidates = [
                build_dir / "diffusion_sample",
                build_dir / "diffusion_sample.exe",
                build_dir / "Release" / "diffusion_sample.exe",
            ]
            example = next((path for path in candidates if path.is_file()), None)
            if example is not None:
                subprocess.run([str(example), str(converted), "5"], check=True)
            benchmark_candidates = [
                build_dir / "flowedge_diffusion_latency_bench",
                build_dir / "flowedge_diffusion_latency_bench.exe",
                build_dir / "Release" / "flowedge_diffusion_latency_bench.exe",
            ]
            benchmark = next((path for path in benchmark_candidates if path.is_file()), None)
            if benchmark is not None:
                subprocess.run([str(benchmark), str(converted), "5", "5"], check=True)


if __name__ == "__main__":
    main()
