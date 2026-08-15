"""One-off export: load the trained Scout Mini PPO checkpoint and export its actor
to ONNX for deployment, using ActorCriticSRU's own real export_onnx() method (NOT
isaaclab_rl's generic exporter - that one doesn't capture the attention/fusion stage,
confirmed by direct inspection of an exported file's input shape). export_onnx's own
docstring confirms it captures the complete pipeline: CrossAttention -> MemorySRU ->
LinearDropout -> Actor MLP.

Does NOT build a full IsaacLab env, and does NOT need the Kit app / AppLauncher -
rsl_rl.modules has no isaaclab dependency, confirmed by this script itself running
as plain Python. Writes to a SCRATCH location only, not the real deployment repo -
per the "copy first, don't edit in place" rule, the output gets reviewed and only
copied into sru-robot-deployment afterward, deliberately, not as a side effect of
this script.
"""

import torch

from rsl_rl.modules import ActorCriticSRU

CHECKPOINT = "/media/user/data1/joshua/IsaacLab/logs/rsl_rl/scout_mini_navigation_ppo/2026-08-15_14-30-39/model_2999.pt"
OUT_DIR = "/tmp/claude-1005/-home-joshua/16892984-d37d-4be3-b8b6-92046867b7ca/scratchpad/exported_policy"

# Network architecture must match ScoutMiniNavPPORunnerCfg.policy exactly (agent_config/agents/rsl_rl_cfg.py)
policy_kwargs = dict(
    num_actor_obs=2575,   # confirmed real shape from the Stage 2/3 runs' own logs ("num obs 2575")
    num_critic_obs=5712,  # confirmed real shape ("num critic obs 5712")
    num_actions=2,        # Scout Mini: genuinely 2-DOF [v, omega]
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
    init_noise_std=1.0,
    rnn_type="lstm_sru",
    rnn_hidden_size=512,
    rnn_num_layers=1,
    dropout=0.2,
    num_cameras=1,
    image_input_dims=(64, 5, 8),
    height_input_dims=(64, 7, 7),
)

print("[export] constructing ActorCriticSRU...")
actor_critic = ActorCriticSRU(**policy_kwargs)

print(f"[export] loading checkpoint: {CHECKPOINT}")
loaded_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
actor_critic.load_state_dict(loaded_dict["model_state_dict"])
actor_critic.eval()
print("[export] checkpoint loaded OK.")

print(f"[export] exporting ONNX to: {OUT_DIR}/nav_policy.onnx (real export_onnx method, full pipeline)")
actor_critic.export_onnx(path=OUT_DIR, filename="nav_policy.onnx", normalizer=None)
print("[export] DONE.")
