# Copyright (c) 2026, Joshua.
# SPDX-License-Identifier: MIT
#
# Written for this project - not adapted from leggedrobotics' b2w.py/aow_d.py beyond
# following their general ArticulationCfg structure (see assets/b2w_reference.py.txt).
# B2W/AoW-D have leg + wheel joints and a low-level locomotion policy; Scout Mini is
# a plain 4-wheel differential-drive base, so this config is much simpler: no leg
# actuator group, no joint_pos targets, just 4 continuous wheel joints.

"""Configuration for the AgileX Scout Mini robot.

The following configuration parameters are available:

* :obj:`SCOUT_MINI_CFG`: Scout Mini, a 4-wheel differential-drive mobile base.

Reference:
    Scout Mini's real hardware is 2-DOF differential drive (forward velocity + yaw
    rate), unlike the 3-DOF Unitree B2W / ANYmal-on-Wheels this codebase was
    originally built for. See PLAN.md secs 4.1/6/7 for the full reasoning.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Local assets directory for this extension
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

__all__ = ["SCOUT_MINI_CFG"]


# Wheel velocity limit derived from two confirmed real values (PLAN.md sec 6), not a
# guess: max commanded linear velocity (+-3.0 m/s, AgileX CAN protocol) divided by
# measured wheel radius (0.08m) = 37.5 rad/s of wheel spin at that command.
_WHEEL_VELOCITY_LIMIT_RAD_S = 3.0 / 0.08

SCOUT_MINI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_ASSETS_DIR}/Robots/ScoutMini/scout_mini.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=None,
            max_angular_velocity=None,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    # base_link origin is at the chassis box's vertical center (see the URDF header
    # for the derivation); 0.180m is that height above the ground plane given the
    # confirmed ground clearance (115mm) and chassis box height (130mm).
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.180),
        joint_vel={".*_wheel_joint": 0.0},
    ),
    actuators={
        # Single actuator group - all 4 wheels are identical, velocity-controlled.
        # velocity_limit_sim is real (derived above). effort_limit_sim is NOT a real
        # spec - wheel torque/motor rating is not published anywhere found (PLAN.md
        # sec 9, still open) - left generous so it doesn't artificially constrain
        # training, per the fallback already noted as acceptable in that open
        # question, rather than fabricating a number from an unverified source.
        # damping is an RL/sim control gain (not a hardware constant - none of the
        # reference configs' gains are hardware-measured either), chosen as a
        # reasonable middle value consistent with the URDF's own conversion-time
        # joint drive damping (10.0).
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=_WHEEL_VELOCITY_LIMIT_RAD_S,
            stiffness={".*": 0.0},
            damping={".*": 10.0},
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for the AgileX Scout Mini robot (4-wheel differential drive)."""
