#!/usr/bin/env python3
"""Convert a torch / HuggingFace checkpoint into FlowEdge .safetensors layout.

    python convert/convert.py <source> <out.safetensors> [--arch mamba]
        [--component all|backbone|head] [--dtype f32|bf16]

<source> is a .safetensors, .pt, .pth or .bin (a state_dict). 
FlowEdge's tensor convention mirrors HF Mamba (`backbone.*`)
+ an optional `flow.*` action head,
so converting a Mamba checkpoint is a rename + normalize pass.
"""
import argparse
import sys

import torch
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


def _load(path):
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


ARCH = {"mamba": mamba}


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
    args = ap.parse_args()

    sd = ARCH[args.arch](_load(args.source))
    if args.component == "backbone":
        sd = {key: value for key, value in sd.items() if key.startswith("backbone.")}
    elif args.component == "head":
        sd = {key: value for key, value in sd.items() if key.startswith("flow.")}
    if not sd:
        sys.exit("error: no tensors found for the selected architecture and component")

    dt = torch.float32 if args.dtype == "f32" else torch.bfloat16
    sd = {key: _convert_dtype(key, value, dt) for key, value in sd.items()}

    if args.component != "head":
        _require(sd, REQUIRED_BACKBONE, "backbone")
    if args.component == "head" or any(key.startswith("flow.") for key in sd):
        _require(sd, REQUIRED_FLOW, "flow head")

    save_file(sd, args.out)
    layer_ids = [int(key.split(".")[2]) for key in sd if key.startswith("backbone.layers.")]
    layers = 1 + max(layer_ids) if layer_ids else 0
    head = "flow" if any(k.startswith("flow.") for k in sd) else "none"
    print(f"wrote {args.out}: {len(sd)} tensors, {layers} layers, head={head}, dtype={args.dtype}")


if __name__ == "__main__":
    main()
