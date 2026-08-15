# Copyright (c) 2026, Joshua.
# SPDX-License-Identifier: MIT
#
# Scout Mini specific navigation environment configuration. Structure mirrors
# `sru-navigation-sim`'s `config/b2w/navigation_env_cfg.py` (B2WNavigationEnvCfg) -
# same override pattern (subclass the shared NavigationEnvCfg, replace scene/action/
# reward/termination fields in __post_init__) - but the *content* differs
# substantially since Scout Mini has no legs, no low-level locomotion policy, and a
# genuinely 2-DOF action space. See PLAN.md sec 7 for the decisions this follows.

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

from navigation_template.env_config.env_cfg_base import NavigationEnvCfg
import navigation_template.mdp as mdp

from navigation_template.assets import SCOUT_MINI_CFG  # isort: skip


# Real, confirmed values - see PLAN.md sec 6.
WHEEL_RADIUS = 0.08
TRACK_WIDTH = 0.49
MAX_LINEAR_VELOCITY = 3.0
MAX_ANGULAR_VELOCITY = 2.523


@configclass
class ScoutMiniNavigationEnvCfg(NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        from navigation_template.mdp.observations import initialize_depth_noise_generator
        from navigation_template.mdp.depth_utils.camera_config import get_camera_config

        initialize_depth_noise_generator(robot_name="scout_mini", use_jit_precompiled=False)

        camera_config = get_camera_config("scout_mini")

        # --- Robot ---
        self.scene.robot = SCOUT_MINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # --- Depth camera ---
        # Real ZED 2 intrinsics derived from Stereolabs' official datasheet (2.12mm
        # focal length, 2um pixel pitch, 720p = 2x2 binning -> 530px fx at 1280w),
        # scaled to a 192x120 raycast resolution matching the upstream convention
        # (raycast native res -> downsample_factor=3 -> 64x40 to match the VAE
        # encoder's fixed input size). See camera_config.py's ZED2_CAMERA_CONFIG
        # comment for the full derivation and its one remaining caveat (nominal
        # ZED 2 optics, not this specific unit's factory stereo calibration).
        # Real ZED 2 intrinsics derived from Stereolabs' official datasheet (2.12mm
        # focal length, 2um pixel pitch, 720p = 2x2 binning -> 530px fx at 1280w-
        # wide 720p). Raycasting directly at the final 64x40 resolution (matching
        # the VAE encoder's fixed input size, and matching how env_cfg_base.py's
        # own default was fixed - see that file's comment on why
        # `from_ros_camera_info` isn't used: it doesn't exist in this server's
        # installed IsaacLab v2.1.1). fx scaled from the real 1280px-wide value
        # down to 64px the same way (528... /1280*64): 530.0 * 64/1280 = 26.5.
        zed2_fx_64 = 2.12 / (0.002 * 2) * 64 / 1280  # = 26.5
        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # Mounted at the front of the chassis, roughly centered, near the top -
        # not a measured mount position (no camera mount drawing available), a
        # reasonable placement pending a real measurement. 0 degree tilt (level),
        # unlike B2W's 20-degree-down tilt, since Scout Mini's camera should be
        # roughly level for a ground vehicle rather than angled for a legged
        # robot's typical forward-and-down gait view.
        self.scene.raycast_camera.offset.pos = (0.30, 0.0, 0.10)
        self.scene.raycast_camera.offset.rot = (1.0, 0.0, 0.0, 0.0)
        # cx/cy left at image center (32, 20) since no factory-calibrated
        # principal-point offset is available for this specific unit (see
        # camera_config.py's ZED2_CAMERA_CONFIG comment).
        self.scene.raycast_camera.pattern_cfg = patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=[zed2_fx_64, 0.0, 32.0, 0.0, zed2_fx_64, 20.0, 0.0, 0.0, 1.0],
            width=64,
            height=40,
        )

        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # --- Actions: genuinely 2-DOF differential drive, no low-level policy ---
        self.actions.velocity_command = mdp.DifferentialDriveActionCfg(
            asset_name="robot",
            left_joint_names=["front_left_wheel_joint", "rear_left_wheel_joint"],
            right_joint_names=["front_right_wheel_joint", "rear_right_wheel_joint"],
            wheel_radius=WHEEL_RADIUS,
            track_width=TRACK_WIDTH,
            max_linear_velocity=MAX_LINEAR_VELOCITY,
            max_angular_velocity=MAX_ANGULAR_VELOCITY,
        )

        # --- Rewards ---
        self.rewards.joint_acc_l2_joint.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"])
        }
        # Dropped per PLAN.md sec 7 decision 4: rewards/penalizes lateral (y-axis)
        # movement, a DOF Scout Mini's differential drive genuinely does not have -
        # not dead weight left in for "compatibility", removed outright.
        self.rewards.lateral_movement = None

        # --- Observations ---
        # low_level_policy is B2W/AoW-D-specific scaffolding: its `actions` term
        # (mdp.last_low_level_action) reads `action_term.low_level_actions`, which
        # only `PerceptiveNavigationSE2Action` has (it's the hierarchical low-level-
        # policy interface). DifferentialDriveAction has no such concept - there's
        # no low-level policy, `velocity_command` IS the final joint-velocity
        # command. Confirmed via a real smoke-test crash (AttributeError), not
        # something caught by inspection alone - see STAGE_2_REPORT.md.
        self.observations.low_level_policy = None

        # --- Events ---
        # randomize_action_scale directly writes `action_term_obj._policy_scaling`/
        # `._policy_bias` with no hasattr guard (unlike the low-pass-filter and
        # backward-penalty event functions, which do guard and are left enabled -
        # they're safe no-ops on DifferentialDriveAction). Those two attributes are
        # PerceptiveNavigationSE2Action-specific (policy-output rescaling for its
        # hierarchical low-level policy interface) - would crash the same way
        # low_level_policy did above. Disabled for the same reason: doesn't apply
        # to a direct-drive action term.
        self.events.randomize_action_scale = None

        # --- Terminations ---
        # No legs - only base_link contact is a fall/collision signal.
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
            "threshold": 1.0,
        }

        # --- Terrain curriculum (matches Stage 1's SCOUT_MINI_MAZE_TERRAIN_CFG
        # settings - already curriculum=False, difficulty fixed at the harder end,
        # see terrains/maze_config.py) ---
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class ScoutMiniNavigationEnvCfg_DEV(ScoutMiniNavigationEnvCfg):
    """Small terrain grid for fast iteration / smoke testing (Stage 2 checkpoint)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class ScoutMiniNavigationEnvCfg_PLAY(ScoutMiniNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
