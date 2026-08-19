"""Constants and configuration values for RL navigation controller.

Adapted for Scout Mini's genuinely 2-DOF differential-drive action space
[v, omega], replacing B2W's original 3-DOF [vx, vy, omega]. See
DEPLOYMENT_ADAPTATION.md for the full list of changes and why each was made.
"""

# Control parameters
DEFAULT_CONTROL_FREQUENCY = 5.0  # Hz
# Was 0.25/10.0 (B2W/ZED-X's range, inherited unchanged and never migrated).
# Scout Mini was actually trained against ZED2_CAMERA_CONFIG
# (sru_retrain/exts/.../mdp/depth_utils/camera_config.py): min_depth=0.3,
# max_depth=20.0. Confirmed via direct read of that file, not assumed. This
# matters because depth_callback() zeroes anything outside [min,max] - and the
# trained VAE encoder reads 0.0 as "obstacle right here", not "far/unknown"
# (same convention as training's DepthNoise.forward). Real Gazebo frames run
# ~44% Inf-pixel (measured via check_depth_values.py) - at the old 0.25/10.0
# range, all of that plus anything genuinely 10-20m out got zeroed to "close
# obstacle", which the policy was never trained to see that way.
DEFAULT_MIN_DEPTH = 0.3  # meters
DEFAULT_MAX_DEPTH = 20.0  # meters
ARRIVE_GOAL_THRESHOLD = 0.75  # meters
NEAR_GOAL_THRESHOLD_MULTIPLIER = 2.0
JOYSTICK_TIMEOUT = 15.0  # seconds

# Model parameters
# B2W's POLICY_SCALE/LATERAL_VELOCITY_SCALE are gone: those existed because B2W's
# policy output was tanh-squashed to [-1, 1] and needed rescaling to real units, and
# had a genuine lateral (y) component. Scout Mini's DifferentialDriveAction (see
# sru_retrain/mdp/navigation/actions/differential_drive_action.py) already outputs
# real-unit [v, omega] directly (scale/offset then a hard clamp, no tanh) - so
# deployment needs to replicate that clamp, not a tanh+scale.
# Real, confirmed values - PLAN.md sec 6/7 decision 6, AgileX CAN protocol.
MAX_LINEAR_VELOCITY = 3.0  # m/s
MAX_ANGULAR_VELOCITY = 2.523  # rad/s

# Filter coefficients - now 2-dim [linear_x, angular_z], no lateral component
# (Scout Mini physically cannot strafe - differential drive, not omnidirectional).
LOW_PASS_FILTER_COEF = [0.9, 0.5]

# Joystick axis mappings
JOYSTICK_AXIS_LINEAR_X = 1
JOYSTICK_AXIS_LINEAR_Y = 0
JOYSTICK_AXIS_LINEAR_Z = 3
JOYSTICK_AXIS_ANGULAR_Z = 2
JOYSTICK_AXIS_SMART = 5

# Joystick button mappings
BUTTON_RESET_HIDDEN_STATE = 2
BUTTON_RECORD_WAYPOINT = 6
BUTTON_CLEAR_WAYPOINT = 4
BUTTON_SEND_GOAL = 10
BUTTON_ABORT = 9
BUTTON_TRIGGER_WAYPOINTS = 1
BUTTON_FORWARD = 11
BUTTON_BACKWARD = 12
BUTTON_LEFT = 13
BUTTON_RIGHT = 14
BUTTON_UP = 3
BUTTON_DOWN = 0

# Scales
LINEAR_SCALE = 1.0
ANGULAR_SCALE = 1.0
MOVING_SCALE = 0.2
SMART_JOYSTICK_SCALE = 5.0
SMART_JOYSTICK_UPDATE_FREQUENCY = 5.0  # Hz
SMART_JOYSTICK_Z_SCALE = 0.25  # Reduce Z movement
SMART_JOYSTICK_FILTER_ALPHA = 0.2  # Low-pass filter coefficient

# Timer intervals
WAYPOINT_PUBLISH_INTERVAL = 0.2  # 5 Hz
TARGET_VECTOR_PUBLISH_INTERVAL = 0.2  # 5 Hz
TRIGGER_BUTTON_COOLDOWN = 1.0  # seconds

# Visualization parameters
TWIST_MARKER_SCALE = 5.0
TWIST_MARKER_ID = 0
TARGET_VECTOR_MARKER_ID = 1
MOVING_GOAL_MARKER_ID = 2
WAYPOINTS_MARKER_ID = 3

# Gravity constant
GRAVITY_MAGNITUDE = 9.81  # m/s^2
