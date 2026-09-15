#!/usr/bin/env python3
"""Export an upstream SmolVLA VLM-cache/action-expert parity capture.

This command accepts only a real, batch-one LeRobot source-pipeline capture.
It does not invent observations, language tokens, flow noise, or timesteps.
The resulting NPZ is the input to ``verify_smolvla_cached_expert.py`` and is
not a latency or task-quality artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


DEFAULT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
EXPECTED_EXPERT_LAYERS = 16
EXPECTED_KEY_VALUE_WIDTH = 320


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_digest(array) -> str:
    value = hashlib.sha256()
    value.update(str(array.dtype).encode("ascii"))
    value.update(json.dumps(list(array.shape)).encode("ascii"))
    value.update(array.tobytes(order="C"))
    return value.hexdigest()


def required(capture, name: str, dtype, shape: tuple[int, ...]):
    if name not in capture:
        raise ValueError(f"capture is missing {name!r}")
    value = capture[name]
    if value.dtype != dtype:
        raise ValueError(f"capture {name!r} must be {dtype}, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"capture {name!r} must have shape {shape}, got {tuple(value.shape)}")
    return value


def cpu_f32(tensor, torch):
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local lerobot/smolvla_base directory")
    parser.add_argument("capture", type=Path, help="real source-pipeline .npz capture")
    parser.add_argument("--output", type=Path, required=True, help="write expert reference NPZ")
    parser.add_argument("--manifest", type=Path, help="write JSON provenance")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()

    try:
        import lerobot
        import numpy as np
        import torch
        import transformers
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks

        if not args.capture.is_file():
            raise ValueError(f"capture does not exist: {args.capture}")
        policy = SmolVLAPolicy.from_pretrained(args.model).eval()
        model = policy.model
        device = next(policy.parameters()).device
        batch_size = 1
        state_dim = policy.config.state_feature.shape[0]
        capture = np.load(args.capture, allow_pickle=False)
        state = required(capture, "observation.state", np.dtype("float32"), (batch_size, state_dim))
        tokens = required(
            capture,
            "observation.language.tokens",
            np.dtype("int64"),
            (batch_size, policy.config.tokenizer_max_length),
        )
        language_mask = required(
            capture,
            "observation.language.attention_mask",
            np.dtype("int64"),
            (batch_size, policy.config.tokenizer_max_length),
        )
        noisy_actions = required(
            capture,
            "noisy_actions",
            np.dtype("float32"),
            (batch_size, policy.config.chunk_size, policy.config.max_action_dim),
        )
        timestep = required(capture, "timestep", np.dtype("float32"), (batch_size,))
        if (
            not np.isfinite(state).all()
            or not np.isfinite(noisy_actions).all()
            or not np.isfinite(timestep).all()
        ):
            raise ValueError("capture state, noisy_actions, and timestep must be finite")
        if not np.isin(language_mask, (0, 1)).all():
            raise ValueError("capture language attention mask must contain only 0 or 1")
        batch = {
            "observation.state": torch.from_numpy(state).to(device),
            "observation.language.tokens": torch.from_numpy(tokens).to(device),
            "observation.language.attention_mask": torch.from_numpy(language_mask).to(device),
        }
        for key, feature in policy.config.image_features.items():
            channels, height, width = feature.shape
            image = required(capture, key, np.dtype("float32"), (batch_size, channels, height, width))
            if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
                raise ValueError(f"capture {key!r} must be finite RGB in [0, 1]")
            batch[key] = torch.from_numpy(image).to(device)

        with torch.no_grad():
            images, image_masks = policy.prepare_images(batch)
            prepared_state = policy.prepare_state(batch)
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images,
                image_masks,
                batch["observation.language.tokens"],
                batch["observation.language.attention_mask"],
                state=prepared_state,
            )
            prefix_attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, cache = model.vlm_with_expert.forward(
                attention_mask=prefix_attention,
                position_ids=prefix_positions,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
                fill_kv_cache=True,
            )
            if not isinstance(cache, dict) or len(cache) != EXPECTED_EXPERT_LAYERS:
                raise RuntimeError("source VLM did not produce the expected 16-layer K/V cache")
            suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(
                torch.from_numpy(noisy_actions).to(device), torch.from_numpy(timestep).to(device)
            )
            suffix_length = suffix_pad_masks.shape[1]
            prefix_length = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_length, prefix_length
            )
            suffix_attention = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_attention = torch.cat([prefix_pad_2d_masks, suffix_attention], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            suffix_positions = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            outputs, _ = model.vlm_with_expert.forward(
                attention_mask=full_attention,
                position_ids=suffix_positions,
                past_key_values=cache,
                inputs_embeds=[None, suffix_embs],
                use_cache=True,
                fill_kv_cache=False,
            )
            source_hidden = outputs[1][:, -policy.config.chunk_size :].to(dtype=torch.float32)
            source_velocity = model.action_out_proj(source_hidden)

        prefix_keys = []
        prefix_values = []
        for layer in range(EXPECTED_EXPERT_LAYERS):
            layer_cache = cache.get(layer)
            if not isinstance(layer_cache, dict):
                raise RuntimeError(f"source VLM cache is missing layer {layer}")
            keys_tensor = layer_cache.get("key_states")
            values_tensor = layer_cache.get("value_states")
            if keys_tensor is None or values_tensor is None:
                raise RuntimeError(f"source VLM cache layer {layer} has no key/value tensors")
            keys = cpu_f32(keys_tensor, torch)
            values = cpu_f32(values_tensor, torch)
            expected_cache_shape = (batch_size, prefix_length, 4, 80)
            if keys.shape != expected_cache_shape or values.shape != expected_cache_shape:
                raise RuntimeError(
                    f"source VLM cache layer {layer} has {keys.shape}/{values.shape}, "
                    f"expected {expected_cache_shape}"
                )
            prefix_keys.append(keys.reshape(prefix_length, EXPECTED_KEY_VALUE_WIDTH))
            prefix_values.append(values.reshape(prefix_length, EXPECTED_KEY_VALUE_WIDTH))

        prefix_mask = cpu_f32(prefix_pad_masks, torch).astype(np.uint8, copy=False)[0]
        expected_mask = np.r_[
            np.ones(prefix_mask.sum(), dtype=np.uint8),
            np.zeros(prefix_length - prefix_mask.sum(), dtype=np.uint8),
        ]
        if not np.array_equal(prefix_mask, expected_mask):
            raise RuntimeError("source prefix mask is not leading valid tokens followed by padding")
        output = {
            "noisy_actions": noisy_actions[0],
            "timestep": timestep,
            "prefix_keys": np.stack(prefix_keys),
            "prefix_values": np.stack(prefix_values),
            "prefix_mask": prefix_mask,
            "expected_hidden": cpu_f32(source_hidden, torch)[0],
            "expected_velocity": cpu_f32(source_velocity, torch)[0],
        }
        expected_hidden_shape = (policy.config.chunk_size, model.vlm_with_expert.expert_hidden_size)
        expected_velocity_shape = (policy.config.chunk_size, policy.config.max_action_dim)
        if output["expected_hidden"].shape != expected_hidden_shape or output["expected_velocity"].shape != expected_velocity_shape:
            raise RuntimeError("source action-expert output has an unexpected shape")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **output)
        report = {
            "schema_version": 1,
            "model": "lerobot/smolvla_base",
            "revision": args.revision,
            "source_capture_sha256": digest(args.capture),
            "source_noisy_actions_sha256": array_digest(output["noisy_actions"]),
            "source_prefix_keys_sha256": array_digest(output["prefix_keys"]),
            "source_prefix_values_sha256": array_digest(output["prefix_values"]),
            "source_hidden_sha256": array_digest(output["expected_hidden"]),
            "source_velocity_sha256": array_digest(output["expected_velocity"]),
            "prefix_length": int(prefix_length),
            "expert_layers": EXPECTED_EXPERT_LAYERS,
            "key_value_width": EXPECTED_KEY_VALUE_WIDTH,
            "expert_hidden_shape": list(output["expected_hidden"].shape),
            "velocity_shape": list(output["expected_velocity"].shape),
            "lerobot_version": lerobot.__version__,
            "transformers_version": transformers.__version__,
            "status": "passed",
            "scope": (
                "upstream SmolVLA captured VLM K/V cache and single action-expert velocity from "
                "a supplied real observation; not FlowEdge parity, latency, or policy-quality evidence"
            ),
        }
    except (ImportError, OSError, RuntimeError, ValueError, StopIteration) as error:
        print(f"SmolVLA action-expert reference export failed: {error}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
