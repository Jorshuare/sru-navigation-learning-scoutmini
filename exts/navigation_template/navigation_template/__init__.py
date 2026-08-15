##
# Apply patches to Isaac Lab's terrain system before anything else.
#
# This is REQUIRED, not optional decoration: patches.py adds height-field storage
# to TerrainGenerator/TerrainImporter that the goal-command generator depends on
# (confirmed for real - omitting this call produces
# `ValueError: No height field data found on terrain...` at env.reset(), caught by
# Stage 2's smoke test, not by inspection). Loaded via direct file import rather
# than `from .terrains import apply_terrain_patches`, matching
# `sru-navigation-sim`'s own top-level `__init__.py` - importing through
# `terrains/__init__.py` normally would trigger that package's other imports
# (terrain generation code) before the patch is actually applied, which is exactly
# the ordering this must avoid.
##
import importlib.util as _importlib_util
import os as _os

_patches_path = _os.path.join(_os.path.dirname(__file__), "terrains", "patches.py")
_spec = _importlib_util.spec_from_file_location("patches", _patches_path)
_patches_module = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_patches_module)
_patches_module.apply_terrain_patches()
del _patches_path, _spec, _patches_module

# This registers the Gym environments via the __init__.py file in the agent_config directory.
from .agent_config import *  # noqa: F401, F403

# Environment configurations
from .env_config import *  # noqa: F401, F403