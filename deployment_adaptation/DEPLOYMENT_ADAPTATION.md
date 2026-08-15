# Scout Mini deployment adaptation

Adapts `sru-robot-deployment`'s `rl_nav_controller` (built for B2W's 3-DOF
`[vx, vy, omega]` action space) for Scout Mini's genuine 2-DOF differential-drive
`[v, omega]` policy trained in this project (checkpoint:
`scout_mini_navigation_ppo/2026-08-15_14-30-39/model_2999.pt`).

Everything here was built and verified inside `sru_retrain/deployment_adaptation/`
first - nothing in `sru-robot-deployment` has been touched yet. `original_backup/`
holds untouched copies of every file this would replace.

## What changed and why

1. **The exporter (`isaaclab_rl`'s generic ONNX exporter) doesn't support this
   architecture at all.** It only wraps the RNN + final MLP, with no concept of the
   attention/fusion stage `ActorCriticSRU` uses ("Self-attention → Cross-attention →
   SRU → MLP" per its own docstring). Confirmed by inspecting an export's actual
   input shape (79, not ~2576) before finding the right tool.
   - The real, official tool is `ActorCriticSRU.export_onnx()` - already built into
     the installed `rsl_rl` fork, found via a commit in the local
     `sru-navigation-sim` clone ("Update navigation config and add ONNX export
     support"). Its own docstring confirms it captures the complete pipeline.
   - `export_scout_mini_policy.py` uses this real method. No IsaacLab/Kit app
     needed - `rsl_rl.modules` has no isaaclab dependency.
   - An earlier attempt patched `isaaclab_rl`'s generic exporter directly (to
     recognize `lstm_sru`) before this was found - that patch has been reverted
     (`git checkout`), since it turned out to solve the wrong problem.

2. **`STATE_DIM`: 16 → 15.** B2W's state vector included a 3-dim `last_action`
   (`vx, vy, omega`); Scout Mini's is 2-dim (`v, omega`). Confirmed against the real
   exported model's own `obs` input shape: `(1, 2575)`, not `(1, 2576)`.

3. **No `tanh` + `POLICY_SCALE` rescaling.** B2W's policy output was `tanh`-squashed
   to `[-1, 1]` and needed rescaling to real units. Scout Mini's
   `DifferentialDriveAction` (see `sru_retrain/mdp/navigation/actions/`) already
   outputs real-unit `[v, omega]` directly from the network (scale/offset default to
   identity, then a hard clamp - no `tanh`). Deployment now replicates that clamp
   directly: `MAX_LINEAR_VELOCITY = 3.0` m/s, `MAX_ANGULAR_VELOCITY = 2.523` rad/s
   (AgileX CAN protocol limits, `PLAN.md` sec 6/7 decision 6).

4. **No lateral velocity component.** Scout Mini is a differential-drive robot and
   physically cannot strafe, unlike B2W. `LATERAL_VELOCITY_SCALE` is gone entirely
   (not just zeroed). `twist.linear.y` is left as joystick-only (no policy
   contribution) - the joystick teleop path itself is untouched, since it's a
   separate, already-validated control path from Part A of this project.

5. **`vae_encoder.onnx` is reused as-is, not re-exported.** Scout Mini's training
   still uses the same depth VAE encoder weights as B2W/ZED-X (`vae_pretrain_new.pth`
   - not yet retrained for ZED 2, a separate open item tracked in `PLAN.md` sec 9).

## Verification performed

- Real inference smoke test on the exported ONNX file directly (random input,
  checked for finite output, correct shapes) - not just shape inspection.
- `verify_predict_pipeline.py`: end-to-end run of the adapted `predict()` logic
  against the *real* `vae_encoder.onnx` + newly-exported `nav_policy.onnx`, checking
  every intermediate shape (depth embedding 2560, target log 4, state 15, obs 2575,
  action 2), clamping correctness, and that the recurrent hidden state genuinely
  changes the output step-to-step (not a static/broken graph). All passed.
- **Not yet done:** actually running the adapted `rl_nav_controller.py` inside a real
  ROS2 node against the Gazebo Scout Mini simulation. `verify_predict_pipeline.py`
  reimplements `LearningModel`'s logic standalone specifically to avoid needing a
  full ROS2 environment for this first pass - the two must be kept in sync by hand;
  if `rl_nav_controller.py` changes, update the verification script to match, or
  re-derive it by hand from the real file at review time.

## What's still not copied into `sru-robot-deployment`

Nothing yet. `rl_nav_controller.py`, `constants.py`, and `exported_policy/nav_policy.onnx`
in this directory are the finished, verified candidates - copying them into the real
package (replacing the current B2W-era files, which stay backed up in
`original_backup/`) is a deliberate next step, not done automatically.
