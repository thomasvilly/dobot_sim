"""
train.py
Train the Dobot pick-and-place policy using rl_games PPO via Isaac Lab.

Usage:
    isim.bat train.py --num_envs 64 --headless --max_iter 3000
    isim.bat train.py --num_envs 8                              # with viewer
"""
import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Dobot pick-and-place")
parser.add_argument("--num_envs",   type=int,  default=64)
parser.add_argument("--max_iter",   type=int,  default=3000)
parser.add_argument("--checkpoint", type=str,  default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything after this point runs inside the Isaac Sim environment."""

import yaml
import gymnasium as gym
from datetime import datetime

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

import dobot_env_register  # noqa: F401 — registers Isaac-Dobot-Pick-Place-v0

from dobot_env_cfg import DobotEnvCfg


def main():
    # 1. Build env
    env_cfg = DobotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Isaac-Dobot-Pick-Place-v0", cfg=env_cfg)

    # 2. Wrap for rl_games
    rl_device = "cuda:0"
    env = RlGamesVecEnvWrapper(env, rl_device=rl_device, clip_obs=10.0, clip_actions=1.0)

    vecenv.register(
        "ISAACLAB",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "ISAACLAB", "env_creator": lambda **kwargs: env},
    )

    # 3. Load YAML
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_games_ppo_cfg.yaml")
    with open(cfg_path) as f:
        rl_games_cfg = yaml.safe_load(f)

    rl_games_cfg["params"]["config"]["max_epochs"]  = args_cli.max_iter
    rl_games_cfg["params"]["config"]["num_actors"]  = args_cli.num_envs

    # Fix minibatch so total_steps % minibatch == 0
    horizon     = rl_games_cfg["params"]["config"]["horizon_length"]
    total_steps = args_cli.num_envs * horizon
    minibatch   = rl_games_cfg["params"]["config"]["minibatch_size"]
    while total_steps % minibatch != 0:
        minibatch //= 2
    rl_games_cfg["params"]["config"]["minibatch_size"] = minibatch
    print(f"[train] num_envs={args_cli.num_envs}, horizon={horizon}, "
          f"total_steps={total_steps}, minibatch={minibatch}")

    # Log dir
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir  = os.path.join("logs", "rl_games", "dobot_pick_place", run_name)
    os.makedirs(log_dir, exist_ok=True)
    rl_games_cfg["params"]["config"]["train_dir"]             = os.path.abspath(log_dir)
    rl_games_cfg["params"]["config"]["full_experiment_name"]  = run_name

    if args_cli.checkpoint:
        rl_games_cfg["params"]["load_checkpoint"] = True
        rl_games_cfg["params"]["load_path"]       = args_cli.checkpoint

    # 4. Train
    runner = Runner(IsaacAlgoObserver())
    runner.load(rl_games_cfg)
    runner.reset()
    runner.run({"train": True, "play": False, "sigma": None})

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
