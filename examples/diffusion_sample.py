#!/usr/bin/env python3
"""Sample an action horizon from a converted LeRobot Diffusion Policy head."""

import argparse

import numpy as np

import flowedge


parser = argparse.ArgumentParser()
parser.add_argument("model", help="FlowEdge-converted Diffusion Policy safetensors")
parser.add_argument("--steps", type=int, default=10)
args = parser.parse_args()

engine = flowedge.Engine(args.model)
condition = np.zeros(engine.condition_dim, dtype=np.float32)
noise = np.random.default_rng(7).standard_normal(
    (engine.action_horizon, engine.action_dim), dtype=np.float32
)
actions = engine.sample_diffusion(condition, noise, args.steps, "ddim")
print(f"action_horizon={engine.action_horizon} action_dim={engine.action_dim}")
print(actions)
