"""
record.py
Run a trained policy and save episodes as HDF5 for SFT.

Usage:
    isim.bat record.py --checkpoint logs/dobot_ppo/run_XXX/model_3000.pt --num_episodes 200 --enable_cameras

The HDF5 format exactly matches the real-robot dataset:
    observations/images/top  (N, 480, 640, 3) uint8
    observations/state       (N, 7) float32  [x_mm, y_mm, z_mm, 0, 0, 0, gripper]
    action                   (N, 7) float32  [dx_mm, dy_mm, dz_mm, 0, 0, 0, d_grip]
    attrs['instruction']     str
"""
from __future__ import annotations

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",   type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=200)
parser.add_argument("--out_dir",      type=str, default="dataset_hdf5/sim_ppo")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
AppLauncher(args).app

import h5py
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from dobot_env_cfg import DobotEnvCfg
from dobot_env     import DobotEnv
from train         import DobotPPORunnerCfg


INSTRUCTION = "pick up the blue block and put it on the plate"


def save_episode(frames, out_dir, ep_id):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"episode_{ep_id:03d}.h5")
    with h5py.File(fname, "w") as f:
        f.create_dataset(
            "observations/images/top",
            data=np.array([fr["rgb"] for fr in frames], dtype=np.uint8),
            compression="gzip",
        )
        f.create_dataset(
            "observations/state",
            data=np.array([fr["state"] for fr in frames], dtype=np.float32),
        )
        f.create_dataset(
            "action",
            data=np.array([fr["action"] for fr in frames], dtype=np.float32),
        )
        f.attrs["instruction"] = INSTRUCTION
    print(f"  saved {fname}  ({len(frames)} frames)")


def main():
    env_cfg = DobotEnvCfg()
    env_cfg.scene.num_envs = 1  # record one env at a time for clean episodes
    env = DobotEnv(cfg=env_cfg)

    runner_cfg = DobotPPORunnerCfg()
    runner = OnPolicyRunner(env, runner_cfg, log_dir="/tmp/record_tmp", device=env.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    os.makedirs(args.out_dir, exist_ok=True)
    ep_id   = len(glob.glob(os.path.join(args.out_dir, "episode_*.h5"))) + 1
    ep_done = 0

    obs_dict, _ = env.reset()
    obs         = obs_dict["policy"]

    frames      = []
    prev_state  = None

    print(f"[record] Starting. Target: {args.num_episodes} episodes → {args.out_dir}")

    while ep_done < args.num_episodes:
        with torch.no_grad():
            actions = policy(obs)

        # Step
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        # Extract frame data for env 0
        rgb   = env.camera.data.output["rgb"][0, :, :, :3].cpu().numpy()   # (H, W, 3) uint8
        state = env._get_proprio_sdk()[0].cpu().numpy()                      # (7,) float32

        # Action in SDK mm space: convert world-metre delta → mm delta
        act_world = actions[0].cpu().numpy()                                 # [dx, dy, dz, d_grip]
        act_mm    = np.array([
            act_world[0] * 1000.0,
            act_world[1] * 1000.0,
            act_world[2] * 1000.0,
            0.0, 0.0, 0.0,
            act_world[3],
        ], dtype=np.float32)

        frames.append({"rgb": rgb, "state": state, "action": act_mm})

        done = terminated[0] or truncated[0]
        if done:
            if len(frames) > 10:   # skip very short episodes
                save_episode(frames, args.out_dir, ep_id)
                ep_id   += 1
                ep_done += 1
                print(f"  Episode {ep_done}/{args.num_episodes} done ({len(frames)} frames)")
            frames = []

    env.close()
    print(f"[record] Done. {ep_done} episodes saved to {args.out_dir}")


if __name__ == "__main__":
    main()
