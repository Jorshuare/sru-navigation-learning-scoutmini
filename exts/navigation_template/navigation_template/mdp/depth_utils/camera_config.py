# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Camera configuration parameters for different robots and camera types.

This module provides camera-specific parameters for depth noise generation and encoding.
"""

import os
from typing import Optional

from isaaclab.utils import configclass

# Local assets directory for this extension
# Path: depth_utils -> mdp -> navigation -> assets/data
_ASSETS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "data")
)


def _get_encoder_path(model_filename: str) -> str:
    """Helper function to construct encoder model path.

    Args:
        model_filename: Name of the encoder model file (e.g., 'vae_pretrain_new.pth')

    Returns:
        Full path to the encoder model file
    """
    return os.path.join(_ASSETS_DIR, "Policies", "depth_encoder", model_filename)


@configclass
class CameraConfig:
    """Configuration class for camera parameters.

    This class contains all camera-specific parameters needed for depth noise generation
    and depth encoder initialization.
    """

    # Camera intrinsic parameters
    focal_length: float = 25.0
    baseline: float = 0.12

    # Depth range parameters
    min_depth: float = 0.25
    max_depth: float = 10.0

    # Camera resolution (width, height)
    resolution: tuple[int, int] = (53, 30)

    # Depth encoder model path
    depth_encoder_path: str = ""

    def __post_init__(self):
        """Post-initialization to set default encoder path if not provided."""
        if not self.depth_encoder_path:
            self.depth_encoder_path = _get_encoder_path("vae_pretrain_fuse.pth")


# Predefined camera configurations
ZEDX_CAMERA_CONFIG = CameraConfig(
    focal_length=25.0,
    baseline=0.12,
    min_depth=0.25,
    max_depth=10.0,
    resolution=(64, 40),
    depth_encoder_path=_get_encoder_path("vae_pretrain_new.pth"),
)
"""Configuration for ZedX camera (used with b2w and aow_d robots)."""

# ZED 2 camera (Scout Mini's real hardware - confirmed NOT the same as ZED-X).
# Values sourced from Stereolabs' official ZED 2 datasheet where confirmed; one
# field is intentionally left as a flagged placeholder rather than guessed - see
# the TODO below.
#
# Confirmed from the official datasheet:
#   - baseline: 120mm (identical to ZED-X's value here, coincidentally)
#   - depth range: 0.3m - 20m (vs. ZED-X's 0.25 - 10m used above; real ZED-X specs
#     were never independently verified in this project, only inferred from this
#     repo's config, so this isn't necessarily "ZED 2 range extended" - it's just
#     using ZED 2's own real published numbers instead of inferring from ZED-X's)
#   - real hardware resolution confirmed in use: 720p (1280x720), which the
#     datasheet lists under "binning 2x2 mode" - frame rate not yet confirmed
#     (15/30/60fps all listed as available at 720p)
#
# NOT filled in with confidence - focal_length:
#   Confirmed (via depth_noise_encoder.py's own docstring) that this field is in
#   PIXELS, not mm, and it's used directly in `disparity = focal_length * baseline
#   / depth` inside the depth-noise disparity model. ZED-X's value here (25.0) does
#   NOT match the pixel-focal-length used elsewhere in this same repo for the
#   raycasting camera intrinsics (fx=72.7025 at a 192x120 pre-downsample
#   resolution, per navigation_env_cfg.py) - meaning 25.0 is calibrated against
#   some OTHER, unconfirmed internal resolution specific to the noise model, not
#   directly transferable by unit conversion alone. Guessing a value here risks
#   silently miscalibrating the noise model in a way that would not throw an error
#   - it would just quietly train the encoder on the wrong disparity scale.
#   TODO before this config is used for real training: either (a) trace exactly
#   which resolution depth_noise_encoder.py's disparity computation assumes and
#   derive the correct pixel-focal-length for ZED 2 at 720p from its real
#   2.12mm physical focal length + 2um pixel pitch (720p uses 2x2 binning per the
#   datasheet, i.e. effective 4um pitch), or (b) sample a real ZED 2's
#   /zed_node/rgb/camera_info topic at 720p and read fx directly - the more
#   reliable option since it sidesteps needing to know the internal resolution at
#   all. Placeholder value below is ZED-X's number, clearly not correct for ZED 2 -
#   present only so the dataclass has a valid float, not as a considered estimate.
ZED2_CAMERA_CONFIG = CameraConfig(
    focal_length=25.0,  # TODO: NOT verified for ZED 2 - see note above
    baseline=0.12,
    min_depth=0.3,
    max_depth=20.0,
    resolution=(64, 40),  # VAE encoder input size, not camera-native - unchanged
    depth_encoder_path=_get_encoder_path("vae_pretrain_new.pth"),  # TODO: this
    # encoder was pretrained against ZED-X's noise/FOV profile (see PLAN.md sec
    # 4.5) - reused here as a starting point, not confirmed adequate for ZED 2.
)
"""Configuration for ZED 2 camera (Scout Mini's real hardware). See TODOs above -
focal_length and the reused depth encoder are open items, not settled choices."""

# Default camera configuration
DEFAULT_CAMERA_CONFIG = ZEDX_CAMERA_CONFIG
"""Default camera configuration (ZedX camera settings) - kept as the repo's
original default since B2W/AoW-D reference configs still depend on it."""

# Robot-to-camera mapping
ROBOT_CAMERA_CONFIGS = {
    "b2w": ZEDX_CAMERA_CONFIG,
    "aow_d": ZEDX_CAMERA_CONFIG,
    "scout_mini": ZED2_CAMERA_CONFIG,
}
"""Dictionary mapping robot names to their camera configurations."""


def get_camera_config(robot_name: str, use_default_fallback: bool = False) -> CameraConfig:
    """Get camera configuration for a specific robot.

    Args:
        robot_name: Name of the robot (e.g., 'b2w', 'aow_d')
        use_default_fallback: If True, return DEFAULT_CAMERA_CONFIG when robot not found
                             instead of raising an error (default: False)

    Returns:
        Camera configuration for the specified robot

    Raises:
        KeyError: If robot_name not found and use_default_fallback is False
    """
    if robot_name in ROBOT_CAMERA_CONFIGS:
        return ROBOT_CAMERA_CONFIGS[robot_name]

    if use_default_fallback:
        return DEFAULT_CAMERA_CONFIG

    available_robots = ", ".join(sorted(ROBOT_CAMERA_CONFIGS.keys()))
    raise KeyError(
        f"Robot '{robot_name}' not found in camera configurations. "
        f"Available robots: {available_robots}"
    )
