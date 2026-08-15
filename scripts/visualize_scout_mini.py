# Copyright (c) 2026, Joshua.
# SPDX-License-Identifier: MIT
"""Interactive visual check: spawn Scout Mini in the maze terrain and drive it with
a simple scripted command sequence (forward, then a turn) so the action term and
terrain can be watched directly in the Kit viewport.

This is a correctness/visualization aid, not a training script - no policy is
involved, actions are hardcoded. See Stage 2 of PLAN.md / STAGE_2_REPORT.md for
the corresponding automated (headless) smoke test this was extended from.

Usage (drop --headless to see the viewport; requires a display, e.g. this
project's VNC display :10 - see README.md):

    conda activate isaaclab
    cd /path/to/IsaacLab
    CUDA_VISIBLE_DEVICES=0 DISPLAY=:10.0 ./isaaclab.sh -p \
        /path/to/sru_retrain/scripts/visualize_scout_mini.py --num_envs 1
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize Scout Mini driving in the maze terrain.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments to spawn.")
parser.add_argument(
    "--steps", type=int, default=600, help="Number of environment (policy-rate) steps to run before exiting."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import navigation_template  # noqa: F401 - triggers task registration
from navigation_template.env_config import scout_mini_env_cfg

import gymnasium as gym

cfg = scout_mini_env_cfg.ScoutMiniNavigationEnvCfg_DEV()
cfg.scene.num_envs = args_cli.num_envs
cfg.scene.env_spacing = 4.0

env = gym.make("Isaac-Nav-PPO-ScoutMini-Dev-v0", cfg=cfg)
env.reset()

device = env.unwrapped.device
n = cfg.scene.num_envs

print(f"[visualize] Running {args_cli.steps} steps across {n} env(s). Ctrl+C to stop early.")
print("[visualize] Phase 1 (first 60%): forward. Phase 2 (last 40%): forward + turn.")

phase1_end = int(args_cli.steps * 0.6)

for i in range(args_cli.steps):
    actions = torch.zeros(n, 2, device=device)
    if i < phase1_end:
        actions[:, 0] = 1.0  # v: forward, m/s
        actions[:, 1] = 0.0  # omega
    else:
        actions[:, 0] = 0.8  # slow down a bit while turning
        actions[:, 1] = 1.0  # omega: turn left, rad/s
    env.step(actions)

print("[visualize] Done.")
env.close()
simulation_app.close()
