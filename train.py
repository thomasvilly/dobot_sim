"""
train.py
Train the Dobot pick-and-place policy with PPO via rsl-rl.

Usage (from thomas_sim directory):
    isim.bat train.py --num_envs 64 --headless
    isim.bat train.py --num_envs 16               # with viewer
    isim.bat train.py --num_envs 64 --headless --resume logs/dobot_ppo/run_XXX/model_1000.pt
"""
from __future__ import annotations

import argparse
import os
import sys

# ---- Isaac Lab app must be launched BEFORE any omniverse imports ----
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Dobot pick-and-place with PPO")
parser.add_argument("--num_envs",  type=int,   default=64)
parser.add_argument("--max_iter",  type=int,   default=3000, help="PPO iterations")
parser.add_argument("--resume",    type=str,   default=None, help="Path to checkpoint to resume from")
parser.add_argument("--log_dir",   type=str,   default="logs/dobot_ppo")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
AppLauncher(args).app  # keeps app alive for the duration of the script

# ---- Now safe to import Isaac / torch ----
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

from dobot_env_cfg import DobotEnvCfg
from dobot_env     import DobotEnv


# ---------------------------------------------------------------------------
# rsl-rl runner config
# ---------------------------------------------------------------------------
@configclass
class DobotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24        # rollout steps collected before each PPO update
    max_iterations:    int = args.max_iter
    save_interval:     int = 100
    experiment_name:   str = "dobot_pick_place"
    run_name:          str = ""
    resume:            bool = args.resume is not None
    load_run:          str = args.resume or ""
    load_checkpoint:   str = ""
    logger:            str = "tensorboard"
    log_dir:           str = args.log_dir

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        # Observation is huge (image + proprio).
        # Use a small CNN encoder outside rsl-rl, or flatten with MLP.
        # Here we use a plain MLP — works for proprio-only ablation too.
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Build env config with CLI num_envs
    env_cfg            = DobotEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Headless → disable camera rendering during training to save VRAM
    # (re-enable for play/record)
    if args.headless:
        # Still need camera for obs — just don't render the viewport
        pass

    env = DobotEnv(cfg=env_cfg)

    runner_cfg = DobotPPORunnerCfg()
    runner     = OnPolicyRunner(env, runner_cfg, log_dir=runner_cfg.log_dir, device=env.device)

    if args.resume:
        runner.load(args.resume)
        print(f"[train] Resumed from {args.resume}")

    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
