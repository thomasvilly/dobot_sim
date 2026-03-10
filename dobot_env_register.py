"""
dobot_env_register.py
Register the Dobot pick-and-place environment with Gymnasium so Isaac Lab's
standard rl_games train.py can find it via --task Isaac-Dobot-Pick-Place-v0.

Import this file at the top of train.py AFTER AppLauncher has started.
"""
import gymnasium as gym

gym.register(
    id="Isaac-Dobot-Pick-Place-v0",
    entry_point="dobot_env:DobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dobot_env_cfg:DobotEnvCfg",
        "rl_games_cfg_entry_point": "rl_games_ppo_cfg.yaml",
    },
)
