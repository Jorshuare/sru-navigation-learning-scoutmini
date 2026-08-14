# Copyright (c) 2026, Joshua.
# SPDX-License-Identifier: MIT
#
# Task registration. Follows the `Isaac-Nav-PPO-<Robot>[-Dev|-Play]-v0` naming
# pattern used by `sru-navigation-sim`'s per-robot `config/<robot>/__init__.py`
# files (e.g. Isaac-Nav-PPO-B2W-v0) - see PLAN.md sec 1a/8. This project only
# registers one robot, so task registration lives directly here rather than under
# a nested `config/scout_mini/` package.

import gymnasium as gym

from navigation_template.env_config.navigation_env import NavigationEnv
from navigation_template.env_config import scout_mini_env_cfg
from navigation_template.agent_config import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Nav-PPO-ScoutMini-v0",
    entry_point=NavigationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": scout_mini_env_cfg.ScoutMiniNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ScoutMiniNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-ScoutMini-Dev-v0",
    entry_point=NavigationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": scout_mini_env_cfg.ScoutMiniNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ScoutMiniNavPPORunnerDevCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-ScoutMini-Play-v0",
    entry_point=NavigationEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": scout_mini_env_cfg.ScoutMiniNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ScoutMiniNavPPORunnerCfg,
    },
)
