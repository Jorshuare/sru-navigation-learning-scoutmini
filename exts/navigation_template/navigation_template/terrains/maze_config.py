# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# Modifications for Scout Mini (differential-drive, non-stair-climbing robot).
#
# SPDX-License-Identifier: MIT

"""Configuration for maze terrains, adapted for Scout Mini.

Adapted from sru-navigation-sim's MAZE_TERRAIN_CFG (upstream proportions:
maze 30% / non_maze 20% / stairs 30% / pits 20%).

Changes from upstream, and why:
  - "stairs" sub-terrain removed entirely. Confirmed geometrically impossible for
    Scout Mini: literal 5-step, 0.2m-per-step risers (see terrain_constants.py's
    StairConfig), no ramp alternative. A rigid, non-stair-climbing wheeled robot
    cannot traverse this - training on it would only generate failure-terminations,
    not useful signal. See PLAN.md sec 4.4/7.2.
  - "maze"'s add_stairs_to_maze=True -> False. The upstream maze sub-terrain
    optionally embeds small stair structures inside otherwise-maze cells; same
    impossibility applies, so this is disabled too.
  - The 30% freed from dropping "stairs" is redistributed evenly (+10% each) across
    the three remaining types. This is a simple default, not a strongly-justified
    weighting - PLAN.md sec 9 leaves open whether "maze" (most representative of
    real-world corridor navigation) should be weighted higher instead. Easy to
    change here if revisited.
  - "maze"/"non_maze"/"pits" geometry itself is UNCHANGED from upstream: verified
    in Stage-0-adjacent research that the 2m cell grid and 4m pit-crossing bridges
    are generously scaled relative to Scout Mini's ~0.58m width (see PLAN.md sec
    4.4), so no reason to alter corridor width or obstacle density here.
"""

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .hf_terrains_maze_cfg import HfMazeTerrainCfg

SCOUT_MINI_MAZE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=30.0,  # Border around the entire terrain grid (not per-tile)
    num_rows=6,
    num_cols=30,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.5, 1.0),
    sub_terrains={
        "maze": HfMazeTerrainCfg(
            proportion=0.4,  # 0.3 upstream + 10% redistributed from dropped stairs
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=0.5,
            add_stairs_to_maze=False,  # was True upstream - stairs impossible for Scout Mini
        ),
        "non_maze": HfMazeTerrainCfg(
            proportion=0.3,  # 0.2 upstream + 10% redistributed from dropped stairs
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
        ),
        # "stairs" sub-terrain intentionally omitted - see module docstring.
        "pits": HfMazeTerrainCfg(
            proportion=0.3,  # 0.2 upstream + 10% redistributed from dropped stairs
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
            dynamic_obstacles=True,  # Enables pit/trough generation
        ),
    },
)
"""Maze terrain configuration for Scout Mini navigation tasks (no stairs)."""
