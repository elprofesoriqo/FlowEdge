#!/usr/bin/env python3
"""Convert a torch / HuggingFace checkpoint into FlowEdge .safetensors layout.

    python convert/convert.py <source> <out.safetensors> [--arch mamba|diffusion]
        [--component all|backbone|head] [--dtype f32|bf16]

<source> is a .safetensors, .pt, .pth or .bin (a state_dict). 
FlowEdge's tensor convention mirrors HF Mamba (`backbone.*`)
+ an optional `flow.*` action head,
so converting a Mamba checkpoint is a rename + normalize pass.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

# used to validate output.
REQUIRED_BACKBONE = [
    "backbone.embeddings.weight",
    "backbone.layers.0.mixer.A_log",
    "backbone.layers.0.mixer.conv1d.weight",
    "backbone.layers.0.mixer.x_proj.weight",
    "backbone.norm_f.weight",
]
REQUIRED_FLOW = [
    "flow.in_proj.weight",
    "flow.time_proj.weight",
    "flow.cond_proj.weight",
    "flow.out_proj.weight",
]
REQUIRED_DIFFUSION = [
    "dp.meta",
    "dp.dims",
    "dp.action_min",
    "dp.action_max",
    "dp.te1.w",
    "dp.te2.w",
    "dp.f.o.w",
]


def _load(path):
    path = str(path)
    if path.endswith(".safetensors"):
        return load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "module"):  # unwrap common training wrappers
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
    return obj


def mamba(sd):
    """HF / state-spaces Mamba -> FlowEdge. Names align; normalize the embedding key and
    keep only backbone.*/flow.* (drops lm_head, tokenizer, optimizer state, ...)."""
    out = {}
    for k, v in sd.items():
        k = k.replace("backbone.embedding.weight", "backbone.embeddings.weight")
        if k.startswith("backbone.") or k.startswith("flow."):
            out[k] = v
    return out


def _config_error(message):
    raise ValueError(f"unsupported Diffusion Policy config: {message}")


def _processor_action_stats(path):
    """Load MIN_MAX action statistics from a LeRobot postprocessor sidecar."""
    try:
        processor = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read processor config {path}: {exc}") from exc

    if not isinstance(processor, dict):
        _config_error("processor config must be an object")
    steps = processor.get("steps")
    if not isinstance(steps, list):
        _config_error("processor config steps must be a list")
    step = next(
        (
            candidate
            for candidate in steps
            if isinstance(candidate, dict)
            and candidate.get("registry_name") == "unnormalizer_processor"
        ),
        None,
    )
    if step is None:
        _config_error("processor config has no unnormalizer_processor step")
    step_config = step.get("config")
    if not isinstance(step_config, dict):
        _config_error("unnormalizer_processor config must be an object")
    norm_map = step_config.get("norm_map")
    if not isinstance(norm_map, dict):
        _config_error("unnormalizer_processor norm_map must be an object")
    normalization = norm_map.get("ACTION")
    if normalization != "MIN_MAX":
        _config_error(
            f"processor ACTION normalization must be 'MIN_MAX', got {normalization!r}"
        )
    state_file = step.get("state_file")
    if not isinstance(state_file, str) or not state_file:
        _config_error("unnormalizer_processor state_file is required")
    state_path = Path(state_file)
    if state_path.is_absolute() or ".." in state_path.parts:
        _config_error("processor state_file must stay beside the processor config")
    try:
        stats = _load(path.parent / state_path)
    except (OSError, SafetensorError, ValueError) as exc:
        raise ValueError(
            f"cannot read processor state {path.parent / state_path}: {exc}"
        ) from exc
    action_min = stats.get("action.min")
    action_max = stats.get("action.max")
    if action_min is None or action_max is None:
        _config_error("processor state must contain action.min and action.max")
    return normalization, action_min, action_max


def diffusion(sd, config, processor_stats=None):
    """LeRobot ConditionalUnet1D -> short FlowEdge `dp.*` tensor names.

    The RGB encoder is intentionally excluded: FlowEdge consumes the flattened
    observation condition produced by LeRobot immediately before its U-Net.
    """
    if config is None:
        _config_error("config.json is required (pass --config when it is not beside the model)")
    checks = {
        "type": (config.get("type"), "diffusion"),
        "beta_schedule": (config.get("beta_schedule"), "squaredcos_cap_v2"),
        "prediction_type": (config.get("prediction_type"), "epsilon"),
        "use_film_scale_modulation": (config.get("use_film_scale_modulation"), True),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            _config_error(f"{name} must be {expected!r}, got {actual!r}")
    config_normalization = config.get("normalization_mapping", {}).get("ACTION")
    processor_normalization = processor_stats[0] if processor_stats else None
    if (
        config_normalization
        and processor_normalization
        and config_normalization != processor_normalization
    ):
        _config_error(
            "config and processor ACTION normalization disagree "
            f"({config_normalization!r} vs {processor_normalization!r})"
        )
    normalization = processor_normalization or config_normalization
    if normalization != "MIN_MAX":
        _config_error(f"ACTION normalization must be 'MIN_MAX', got {normalization!r}")

    down_dims = [int(value) for value in config["down_dims"]]
    if not 2 <= len(down_dims) <= 8 or any(value <= 0 for value in down_dims):
        _config_error("down_dims must contain between 2 and 8 positive dimensions")
    action_dim = int(config["output_features"]["action"]["shape"][0])
    horizon = int(config["horizon"])
    action_steps = int(config["n_action_steps"])
    observation_steps = int(config["n_obs_steps"])
    kernel = int(config["kernel_size"])
    groups = int(config["n_groups"])
    timestep_dim = int(config["diffusion_step_embed_dim"])
    train_timesteps = int(config["num_train_timesteps"])
    if action_dim <= 0 or horizon <= 0 or horizon % (1 << (len(down_dims) - 1)):
        _config_error("action dimension/horizon is invalid for the configured U-Net depth")
    if observation_steps <= 0 or observation_steps > horizon:
        _config_error("n_obs_steps must be between 1 and horizon")
    if action_steps <= 0 or action_steps > horizon - observation_steps + 1:
        _config_error("n_action_steps does not fit after the observation prefix")
    if kernel <= 0 or kernel % 2 == 0:
        _config_error("kernel_size must be positive and odd")
    if timestep_dim < 4 or timestep_dim % 2:
        _config_error("diffusion_step_embed_dim must be an even value of at least 4")
    if groups <= 0 or any(value % groups for value in down_dims):
        _config_error("every down dimension must be divisible by n_groups")
    if down_dims[0] % 8:
        _config_error("the first down dimension must be divisible by LeRobot's final 8 groups")
    if train_timesteps <= 0:
        _config_error("num_train_timesteps must be positive")
    clip_range = float(config.get("clip_sample_range", 1.0))
    if not math.isfinite(clip_range) or clip_range <= 0:
        _config_error("clip_sample_range must be finite and positive")

    root = "diffusion.unet"
    first_film = f"{root}.down_modules.0.0.cond_encoder.1.weight"
    if first_film not in sd or sd[first_film].ndim != 2:
        raise ValueError(f"checkpoint is missing required tensor {first_film!r}")
    condition_dim = int(sd[first_film].shape[1]) - timestep_dim
    if condition_dim <= 0:
        _config_error("U-Net FiLM width does not contain an observation condition")

    out = {}

    def put(destination, source, shape):
        tensor = sd.get(source)
        if tensor is None:
            raise ValueError(f"checkpoint is missing required tensor {source!r}")
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"tensor {source!r} has shape {tuple(tensor.shape)}, expected {tuple(shape)}"
            )
        out[destination] = tensor

    def conv(destination, source, out_channels, in_channels, size):
        put(f"{destination}.w", f"{source}.weight", (out_channels, in_channels, size))
        put(f"{destination}.b", f"{source}.bias", (out_channels,))

    def linear(destination, source, out_features, in_features):
        put(f"{destination}.w", f"{source}.weight", (out_features, in_features))
        put(f"{destination}.b", f"{source}.bias", (out_features,))

    def norm(destination, source, channels):
        put(f"{destination}.w", f"{source}.weight", (channels,))
        put(f"{destination}.b", f"{source}.bias", (channels,))

    def residual(destination, source, in_channels, out_channels):
        conv(f"{destination}.c1", f"{source}.conv1.block.0", out_channels, in_channels, kernel)
        norm(f"{destination}.n1", f"{source}.conv1.block.1", out_channels)
        linear(
            f"{destination}.film",
            f"{source}.cond_encoder.1",
            2 * out_channels,
            timestep_dim + condition_dim,
        )
        conv(f"{destination}.c2", f"{source}.conv2.block.0", out_channels, out_channels, kernel)
        norm(f"{destination}.n2", f"{source}.conv2.block.1", out_channels)
        if in_channels != out_channels:
            conv(f"{destination}.res", f"{source}.residual_conv", out_channels, in_channels, 1)

    linear("dp.te1", f"{root}.diffusion_step_encoder.1", timestep_dim * 4, timestep_dim)
    linear("dp.te2", f"{root}.diffusion_step_encoder.3", timestep_dim, timestep_dim * 4)

    for stage, out_channels in enumerate(down_dims):
        in_channels = action_dim if stage == 0 else down_dims[stage - 1]
        residual(f"dp.d{stage}.r0", f"{root}.down_modules.{stage}.0", in_channels, out_channels)
        residual(f"dp.d{stage}.r1", f"{root}.down_modules.{stage}.1", out_channels, out_channels)
        if stage + 1 < len(down_dims):
            conv(f"dp.d{stage}.ds", f"{root}.down_modules.{stage}.2", out_channels, out_channels, 3)

    largest = down_dims[-1]
    for block in range(2):
        residual(f"dp.m{block}", f"{root}.mid_modules.{block}", largest, largest)

    for stage in range(len(down_dims) - 1):
        high = down_dims[-1 - stage]
        low = down_dims[-2 - stage]
        residual(f"dp.u{stage}.r0", f"{root}.up_modules.{stage}.0", 2 * high, low)
        residual(f"dp.u{stage}.r1", f"{root}.up_modules.{stage}.1", low, low)
        # LeRobot uses equal input/output widths here, so its ConvTranspose1d
        # storage [in,out,k] is also [low,low,4].
        conv(f"dp.u{stage}.us", f"{root}.up_modules.{stage}.2", low, low, 4)

    conv("dp.f.c", f"{root}.final_conv.0.block.0", down_dims[0], down_dims[0], kernel)
    norm("dp.f.n", f"{root}.final_conv.0.block.1", down_dims[0])
    conv("dp.f.o", f"{root}.final_conv.1", action_dim, down_dims[0], 1)

    if processor_stats:
        action_min, action_max = processor_stats[1:]
        if tuple(action_min.shape) != (action_dim,) or tuple(action_max.shape) != (action_dim,):
            raise ValueError(
                "processor action statistics must have shape "
                f"({action_dim},), got {tuple(action_min.shape)} and {tuple(action_max.shape)}"
            )
        out["dp.action_min"] = action_min
        out["dp.action_max"] = action_max
    else:
        min_key = "unnormalize_outputs.buffer_action.min"
        max_key = "unnormalize_outputs.buffer_action.max"
        if min_key not in sd or max_key not in sd:
            min_key = "normalize_targets.buffer_action.min"
            max_key = "normalize_targets.buffer_action.max"
        put("dp.action_min", min_key, (action_dim,))
        put("dp.action_max", max_key, (action_dim,))
    action_min = out["dp.action_min"]
    action_max = out["dp.action_max"]
    if not torch.isfinite(action_min).all() or not torch.isfinite(action_max).all():
        _config_error("action normalization statistics must be finite")
    if torch.any(action_max <= action_min):
        _config_error("action normalization max must be strictly greater than min")
    out["dp.dims"] = torch.tensor(down_dims, dtype=torch.float32)
    out["dp.meta"] = torch.tensor(
        [
            1,
            action_dim,
            horizon,
            action_steps,
            observation_steps,
            condition_dim,
            len(down_dims),
            kernel,
            groups,
            timestep_dim,
            train_timesteps,
            0,  # squaredcos_cap_v2
            0,  # epsilon prediction
            int(bool(config.get("clip_sample", True))),
            clip_range,
            0,  # MIN_MAX action normalization
        ],
        dtype=torch.float32,
    )
    return out


ARCH = {"mamba": mamba, "diffusion": diffusion}


def _convert_dtype(name, tensor, weight_dtype):
    """Only 2-D matmul weights may be BF16 in the current runtime."""
    bf16_eligible = (
        tensor.ndim == 2
        and name != "backbone.embeddings.weight"
        and not name.endswith("mixer.A_log")
    )
    dtype = weight_dtype if bf16_eligible else torch.float32
    return tensor.to(dtype).contiguous()


def _require(sd, names, component):
    missing = [name for name in names if name not in sd]
    if missing:
        sys.exit(f"error: {component} is missing required tensors: {missing}")


def main():
    ap = argparse.ArgumentParser(description="Convert a torch/HF checkpoint to FlowEdge .safetensors")
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--arch", choices=sorted(ARCH), default="mamba")
    ap.add_argument("--component", choices=("all", "backbone", "head"), default="all")
    ap.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    ap.add_argument("--config", help="Diffusion Policy config.json (auto-detected beside source)")
    ap.add_argument(
        "--processor",
        help="LeRobot policy_postprocessor.json (auto-detected for a source directory)",
    )
    args = ap.parse_args()

    source = Path(args.source)
    if source.is_dir():
        model_source = source / "model.safetensors"
        default_config = source / "config.json"
    else:
        model_source = source
        default_config = source.with_name("config.json")
    config_path = Path(args.config) if args.config else default_config
    config = None
    processor_stats = None
    if args.arch == "diffusion":
        if args.component == "backbone":
            sys.exit("error: diffusion conversion exports an action head, not a backbone")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            sys.exit(f"error: cannot read diffusion config {config_path}: {exc}")
        processor_path = Path(args.processor) if args.processor else None
        if processor_path is None and source.is_dir():
            candidate = source / "policy_postprocessor.json"
            if candidate.is_file():
                processor_path = candidate
        if processor_path is not None:
            try:
                processor_stats = _processor_action_stats(processor_path)
            except (TypeError, ValueError) as exc:
                sys.exit(f"error: {exc}")
    try:
        loaded = _load(model_source)
        sd = (
            diffusion(loaded, config, processor_stats)
            if args.arch == "diffusion"
            else mamba(loaded)
        )
    except (KeyError, TypeError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    if args.component == "backbone":
        sd = {key: value for key, value in sd.items() if key.startswith("backbone.")}
    elif args.component == "head":
        prefixes = ("flow.", "dp.")
        sd = {key: value for key, value in sd.items() if key.startswith(prefixes)}
    if not sd:
        sys.exit("error: no tensors found for the selected architecture and component")

    dt = torch.float32 if args.dtype == "f32" else torch.bfloat16
    sd = {key: _convert_dtype(key, value, dt) for key, value in sd.items()}

    if args.arch == "mamba" and args.component != "head":
        _require(sd, REQUIRED_BACKBONE, "backbone")
    if args.arch == "diffusion" or args.component == "head" or any(key.startswith("flow.") for key in sd):
        required = REQUIRED_DIFFUSION if args.arch == "diffusion" else REQUIRED_FLOW
        _require(sd, required, f"{args.arch} head")

    save_file(sd, args.out)
    layer_ids = [int(key.split(".")[2]) for key in sd if key.startswith("backbone.layers.")]
    layers = 1 + max(layer_ids) if layer_ids else 0
    head = "diffusion" if any(k.startswith("dp.") for k in sd) else (
        "flow" if any(k.startswith("flow.") for k in sd) else "none"
    )
    print(f"wrote {args.out}: {len(sd)} tensors, {layers} layers, head={head}, dtype={args.dtype}")


if __name__ == "__main__":
    main()
