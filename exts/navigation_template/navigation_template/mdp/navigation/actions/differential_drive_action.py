# Copyright (c) 2026, Joshua.
# SPDX-License-Identifier: MIT
#
# Written for this project. Structure follows IsaacLab's own built-in
# `JointVelocityAction` idiom (isaaclab.envs.mdp.actions.joint_actions) - process
# once per environment step, apply the same target every physics step and let the
# ImplicitActuator's internal PD/velocity controller handle physics-rate tracking -
# rather than B2W's hierarchical `PerceptiveNavigationSE2Action` pattern, which
# doesn't apply here since there's no low-level policy to decimate/run.

"""Differential-drive action term: policy [v, omega] -> per-wheel velocity targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .differential_drive_action_cfg import DifferentialDriveActionCfg


class DifferentialDriveAction(ActionTerm):
    """Maps a 2-DOF [v, omega] command to 4 wheel velocity targets via differential-drive kinematics."""

    cfg: DifferentialDriveActionCfg

    def __init__(self, cfg: DifferentialDriveActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve left/right wheel joints separately - order within each side does
        # not matter, since every wheel on a side gets the identical target velocity.
        self._left_joint_ids, self._left_joint_names = self._asset.find_joints(cfg.left_joint_names)
        self._right_joint_ids, self._right_joint_names = self._asset.find_joints(cfg.right_joint_names)
        self._num_left = len(self._left_joint_ids)
        self._num_right = len(self._right_joint_ids)

        import omni.log

        omni.log.info(
            f"[{self.__class__.__name__}] left wheels: {self._left_joint_names} {self._left_joint_ids}, "
            f"right wheels: {self._right_joint_names} {self._right_joint_ids}"
        )

        self._action_dim = 2  # [v, omega] - genuinely 2-DOF, not padded to 3

        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._scale = torch.tensor(cfg.scale, device=self.device)
        self._offset = torch.tensor(cfg.offset, device=self.device)

        # Per-wheel-side velocity target buffers, reused every apply_actions() call.
        self._left_wheel_vel = torch.zeros(self.num_envs, self._num_left, device=self.device)
        self._right_wheel_vel = torch.zeros(self.num_envs, self._num_right, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Called once per environment (policy) step."""
        self._raw_actions[:] = actions
        processed = self._raw_actions * self._scale + self._offset
        # Hard-clip to Scout Mini's real, confirmed CAN-protocol velocity limits
        # (PLAN.md sec 6/7 decision 6) - the sim actuator can't exceed these anyway
        # (velocity_limit_sim in scout_mini.py is the wheel-level equivalent), but
        # clipping the *commanded* v/omega here keeps the policy's action space
        # itself bounded to physically achievable commands.
        processed[:, 0] = torch.clamp(processed[:, 0], -self.cfg.max_linear_velocity, self.cfg.max_linear_velocity)
        processed[:, 1] = torch.clamp(processed[:, 1], -self.cfg.max_angular_velocity, self.cfg.max_angular_velocity)
        self._processed_actions[:] = processed

    def apply_actions(self):
        """Called every physics step - reapplies the same target (matches
        IsaacLab's own JointVelocityAction idiom; the ImplicitActuator's PD
        controller handles physics-rate tracking of a held target)."""
        v = self._processed_actions[:, 0]
        omega = self._processed_actions[:, 1]

        half_track = self.cfg.track_width / 2.0
        left_speed = (v - omega * half_track) / self.cfg.wheel_radius
        right_speed = (v + omega * half_track) / self.cfg.wheel_radius

        self._left_wheel_vel[:] = left_speed.unsqueeze(-1)
        self._right_wheel_vel[:] = right_speed.unsqueeze(-1)

        self._asset.set_joint_velocity_target(self._left_wheel_vel, joint_ids=self._left_joint_ids)
        self._asset.set_joint_velocity_target(self._right_wheel_vel, joint_ids=self._right_joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
        return {}
