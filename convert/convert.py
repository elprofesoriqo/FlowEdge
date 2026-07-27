#!/usr/bin/env python3
"""Convert a torch / HuggingFace checkpoint into FlowEdge .safetensors layout.

    python convert/convert.py <source> <out.safetensors> [--arch mamba] [--dtype f32|bf16]

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
REQUIRED = [
    "backbone.embeddings.weight",
    "backbone.layers.0.mixer.A_log",
    "backbone.layers.0.mixer.conv1d.weight",
    "backbone.layers.0.mixer.x_proj.weight",
    "backbone.norm_f.weight",
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


def main():
    ap = argparse.ArgumentParser(description="Convert a torch/HF checkpoint to FlowEdge .safetensors")
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--arch", choices=sorted(ARCH), default="mamba")
    ap.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    args = ap.parse_args()

    sd = ARCH[args.arch](_load(args.source))
    if not sd:
        sys.exit("error: no backbone.*/flow.* tensors found — wrong --arch or source?")

    dt = torch.float32 if args.dtype == "f32" else torch.bfloat16
    sd = {k: v.to(dt).contiguous() for k, v in sd.items()}

    missing = [k for k in REQUIRED if k not in sd]
    if missing:
        sys.exit(f"error: converted checkpoint is missing required tensors: {missing}")

    save_file(sd, args.out)
    layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("backbone.layers."))
    head = "flow" if any(k.startswith("flow.") for k in sd) else "none"
    print(f"wrote {args.out}: {len(sd)} tensors, {layers} layers, head={head}, dtype={args.dtype}")


if __name__ == "__main__":
    main()