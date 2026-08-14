# Copyright (c) 2026, Joshua.
# SPDX-License-Identifier: MIT
#
# PPO only for now, per PLAN.md sec 7 decision 5 ("PPO first, with a small
# iteration count... MDPO/DML is a later stage, attempted only once a PPO baseline
# is confirmed to actually learn Scout-appropriate behavior"). Mirrors
# `sru-navigation-sim`'s B2WNavPPORunnerCfg structure/network sizing (same SRU
# architecture, same depth/height feature dims - those are tied to the shared VAE/
# heightscan encoders' fixed output shapes, not robot-specific, so unchanged).

from isaaclab.utils import configclass

from navigation_template.agent_config.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ScoutMiniNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Scout Mini navigation."""

    num_steps_per_env = 16
    max_iterations = 15000
    save_interval = 500
    logger = "wandb"
    seed = 60
    wandb_project = "isaaclab_nav_scout_mini"
    experiment_name = "scout_mini_navigation_ppo"
    empirical_normalization = False
    reward_shifting_value = 0.05
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSRU",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_hidden_size=512,
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        dropout=0.2,
        num_cameras=1,
        image_input_dims=(64, 5, 8),  # depth image: 64 channels * 5 * 8 = 2560 - fixed by the VAE encoder, robot-agnostic
        height_input_dims=(64, 7, 7),  # encoded height_scan_critic: 64*7*7 = 3136 - fixed by the heightscan encoder
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.02,
        use_clipped_value_loss=True,
        clip_param=0.2,
        value_clip_param=0.2,
        entropy_coef=0.00375,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ScoutMiniNavPPORunnerDevCfg(ScoutMiniNavPPORunnerCfg):
    """Development/smoke-test configuration: tiny iteration count, local logging."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3  # Stage 2 checkpoint is "runs end-to-end", not "learns"
        self.save_interval = 1
        self.experiment_name = "scout_mini_navigation_ppo_dev"
        self.logger = "tensorboard"
