# Copyright (c) 2026, Joshua.
# SPDX-License-Identifier: MIT
#
# Scout Mini specific navigation environment configuration. Structure mirrors
# `sru-navigation-sim`'s `config/b2w/navigation_env_cfg.py` (B2WNavigationEnvCfg) -
# same override pattern (subclass the shared NavigationEnvCfg, replace scene/action/
# reward/termination fields in __post_init__) - but the *content* differs
# substantially since Scout Mini has no legs, no low-level locomotion policy, and a
# genuinely 2-DOF action space. See PLAN.md sec 7 for the decisions this follows.

import isaaclab.sim as sim_utils
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

        # --- Ground visual material ---
        # Swapped from the base config's remote MDL marble-tile texture to a plain
        # flat-color material. That texture (TilesMarbleSpiderWhiteBrickBondHoned,
        # fetched live from an NVIDIA Omniverse asset server) has one component -
        # a single-channel/grayscale-encoded JPEG - that reliably fails to load in
        # this server's installed Isaac Sim build: confirmed the file itself is not
        # corrupted (downloaded and inspected it independently, valid JPEG), and
        # confirmed it's not a transient network issue (failed identically on a
        # second full run) - so this looks like a genuine renderer-side encoding
        # incompatibility, not something fixable by retrying or clearing a cache.
        # Purely cosmetic either way: the ground's visual material has no effect on
        # the raycasting depth sensor (computed geometrically against the physics
        # mesh, not the rendered image) or on physics - only on what a human sees
        # when watching a render. A flat color sidesteps needing to fetch/decode
        # this asset at all.
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.6, 0.6, 0.6), roughness=0.8, metallic=0.0
        )

        # --- Depth camera ---
        # Real ZED 2 intrinsics derived from Stereolabs' official datasheet (2.12mm
        # focal length, 2um pixel pitch, 720p = 2x2 binning -> 530px fx at 1280px-
        # wide 720p). Raycasting directly at the final 64x40 resolution (matching
        # the VAE encoder's fixed input size, and matching how env_cfg_base.py's
        # own default was fixed - see that file's comment on why
        # `from_ros_camera_info` isn't used: it doesn't exist in this server's
        # installed IsaacLab v2.1.1). fx scaled from the real 1280px-wide value
        # down to 64px the same way: 530.0 * 64/1280 = 26.5.
        zed2_fx_64 = 2.12 / (0.002 * 2) * 64 / 1280  # = 26.5
        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # Real mount measurements (2026-08-15): camera's lower base sits 19.2cm
        # above the top of the chassis, 6cm forward, facing directly forward with
        # no tilt (confirmed explicitly - "not tilt at any degree").
        #   x: 0.06m forward of base_link (chassis vertical-center) origin - this
        #      is 6cm forward of chassis CENTER, matching how the measurement was
        #      asked for; if it was actually meant as 6cm forward of the chassis's
        #      FRONT EDGE (306mm forward of center) instead, this needs correcting
        #      to x=0.246 - flagged here rather than silently guessed either way.
        #   z: chassis top is 0.065m above base_link (half the 0.130m chassis
        #      height, see scout_mini.urdf's header) + 0.192m mount height above
        #      that = 0.257m above base_link.
        #   y: 0.0 - not specified, assumed centered left-right.
        #   rot: identity - level, forward-facing, no tilt (confirmed, unlike
        #      B2W's 20-degree-down tilt for its legged forward-and-down gait view).
        self.scene.raycast_camera.offset.pos = (0.06, 0.0, 0.257)
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
