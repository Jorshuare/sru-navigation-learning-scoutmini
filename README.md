# SRU → Scout Mini Navigation

An IsaacLab reinforcement-learning navigation policy for the AgileX Scout Mini, based
on the Spatially-enhanced Recurrent Unit (SRU) architecture originally developed for
the Unitree B2W and ANYmal-on-Wheels robots. Scout Mini is a differential-drive
platform (2-DOF: forward velocity + yaw rate), so the policy, action interface, and
terrain are adapted accordingly rather than reused as-is.

## Overview

- **Architecture**: SRU-LSTM recurrent policy with cross-attention fusion of depth
  images and proprioception, trained via PPO/MDPO.
- **Simulation**: Procedurally generated maze/pillar/pit terrains in IsaacLab.
- **Sensing**: Single forward-facing depth camera (ZED 2), no lidar.
- **Action space**: 2-DOF differential-drive velocity command `(v, ω)`.

## Structure

```
sru_retrain/
├── exts/navigation_template/   # IsaacLab extension (terrain, env config, MDP, assets)
├── rsl_rl/                     # SRU network architecture + PPO/MDPO training algorithms
└── scripts/                    # train / play / list_envs entry points
```

## Setup

Requires IsaacLab v2.1.x + Isaac Sim 4.5.0, Python 3.10.

Install as an external IsaacLab extension:

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p -m pip install -e /path/to/sru_retrain/exts/navigation_template
```

Install the training framework:

```bash
./isaaclab.sh -p -m pip install -e /path/to/sru_retrain/rsl_rl
```

## Training

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py --task <task-id> --num_envs <N> --headless
```

On multi-GPU machines, pin to a single GPU explicitly
(`CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh ...`) — the depth-noise model does not
currently handle multiple visible devices correctly.

## Provenance

Built from, and adapted relative to:
- [`leggedrobotics/sru-navigation-sim`](https://github.com/leggedrobotics/sru-navigation-sim)
- [`leggedrobotics/sru-navigation-learning`](https://github.com/leggedrobotics/sru-navigation-learning)
- [`leggedrobotics/sru-pytorch-spatial-learning`](https://github.com/leggedrobotics/sru-pytorch-spatial-learning)
- Extension scaffold from [`leggedrobotics/navigation_template`](https://github.com/leggedrobotics/navigation_template)
- Yang, Frivik, Hoeller, Wang, Cadena, Hutter. *"Spatially-enhanced recurrent memory
  for long-range mapless navigation via end-to-end reinforcement learning."* IJRR 2025.

## License

MIT (see `LICENSE`).
