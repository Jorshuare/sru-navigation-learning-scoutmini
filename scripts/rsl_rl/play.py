"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import navigation_template  # noqa: F401

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    # NOTE: isaaclab_rl's exporter only recognizes standard nn.LSTM/nn.GRU modules
    # (introspected via the module's class name) - it doesn't know about the SRU
    # fork's custom `LSTM_SRU` recurrent module, and raises NotImplementedError for
    # it. That's a real gap in generic deployment export (relevant to Stage 5,
    # sim-to-real prep), not something that should block just watching/evaluating
    # the policy here - confirmed via a real crash during Stage 3 play verification,
    # not assumed. Caught and logged rather than silently skipped, so it's still
    # visible if/when export support actually needs fixing.
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        )
        export_policy_as_onnx(
            ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
    except NotImplementedError as e:
        print(
            f"[WARNING] Skipping JIT/ONNX export - not supported for this policy's RNN type ({e}). "
            "Not needed to play/evaluate the policy; only relevant for deployment export (Stage 5)."
        )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # DIAGNOSTIC (temporary): cross-check whether the raw_omega
    # climb/saturation pattern observed live in Gazebo (raw_omega ramping call
    # to call, well past the deployment-side clamp, e.g. 6.2 -> 8.4 rad/s
    # while the clamped cmd_vel sits pinned at ~2.52) also shows up in a
    # repeated IsaacLab play.py evaluation, independent of Gazebo entirely.
    # `actions` here is the actor's raw mean output straight from the policy,
    # before DifferentialDriveAction.process_actions() applies its own
    # scale/offset/clip inside env.step() - i.e. the IsaacLab-side equivalent
    # of the ONNX runner's `raw_action` (same clip semantics: scale/offset
    # default to identity, so process_actions reduces to a direct clamp).
    # Logs only env index 0 (single-robot equivalent), every step.
    diag_step = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            diag_step += 1
            raw_omega = float(actions[0, 1])
            raw_v = float(actions[0, 0])
            print(
                f'[DIAG play raw_action #{diag_step}] raw_v={raw_v:.6f} '
                f'raw_omega={raw_omega:.6f}',
                flush=True
            )
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
