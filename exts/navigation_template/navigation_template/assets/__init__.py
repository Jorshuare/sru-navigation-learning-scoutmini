# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom robot configurations and assets for navigation tasks."""

import os

# Path to the local data directory containing robots and policies
ISAACLAB_NAV_TASKS_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
"""Path to the navigation tasks assets data directory."""

# NOTE: upstream also exports B2W_CFG/ANYMAL_D_ON_WHEELS_CFG here (`from .b2w import
# *` / `from .aow_d import *`). Not carried forward - this project is Scout-Mini-only
# (see PLAN.md sec 2, scope boundaries); b2w.py/aow_d.py were never copied into this
# package (only assets/b2w_reference.py.txt, kept as a non-importable structural
# reference for writing scout_mini.py). Importing them here would break this
# package's import entirely, since those files - and the B2W/AoW-D USD assets they
# point to - don't exist in this repo.
from .scout_mini import *

__all__ = ["ISAACLAB_NAV_TASKS_ASSETS_DIR", "SCOUT_MINI_CFG"]
