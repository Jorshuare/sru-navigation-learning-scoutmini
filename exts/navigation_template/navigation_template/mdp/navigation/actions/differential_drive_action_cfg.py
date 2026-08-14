# Copyright (c) 2026, Joshua.
# SPDX-License-Identifier: MIT
#
# Written for this project - Scout Mini's real hardware is 2-DOF differential drive
# (forward velocity + yaw rate), not the 3-DOF SE(2) + pretrained-locomotion-policy
# design `PerceptiveNavigationSE2Action` implements for the legged-wheeled B2W/AoW-D
# (see PLAN.md sec 7, decision 1). No low-level policy is needed here - the mapping
# from (v, omega) to wheel velocities is exact differential-drive kinematics, not
# something that needs to be learned.

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .differential_drive_action import DifferentialDriveAction


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """Configuration for the differential-drive action term.

    The policy outputs 2 raw actions ``[v, omega]`` (forward velocity, yaw rate).
    These are scaled/offset, clipped to the robot's real hardware limits, then
    converted to per-wheel angular velocity targets via standard differential-drive
    kinematics:

    .. math::
        \\omega_{left}  = (v - \\omega \\cdot \\text{track\\_width} / 2) / r
        \\omega_{right} = (v + \\omega \\cdot \\text{track\\_width} / 2) / r

    where :math:`r` is the wheel radius.
    """

    class_type: type[ActionTerm] = DifferentialDriveAction

    left_joint_names: list[str] = MISSING
    """Regex(es) matching the left-side wheel joints (e.g. front_left + rear_left)."""

    right_joint_names: list[str] = MISSING
    """Regex(es) matching the right-side wheel joints (e.g. front_right + rear_right)."""

    wheel_radius: float = MISSING
    """Wheel radius in meters. Real measured value for Scout Mini: 0.08m (PLAN.md sec 6)."""

    track_width: float = MISSING
    """Left-right wheel center distance in meters. Real measured value for Scout
    Mini: 0.49m (PLAN.md sec 6)."""

    max_linear_velocity: float = MISSING
    """Hard clip on commanded forward velocity, in m/s. Real confirmed value for
    Scout Mini: 3.0 m/s (AgileX CAN protocol, PLAN.md sec 6/7 decision 6)."""

    max_angular_velocity: float = MISSING
    """Hard clip on commanded yaw rate, in rad/s. Real confirmed value for Scout
    Mini: 2.523 rad/s (AgileX CAN protocol, PLAN.md sec 6/7 decision 6)."""

    scale: list[float] = [1.0, 1.0]
    """Scale applied to the raw [v, omega] actions before clipping."""

    offset: list[float] = [0.0, 0.0]
    """Offset applied to the raw [v, omega] actions before clipping."""
