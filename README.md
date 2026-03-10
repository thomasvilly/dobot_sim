# Dobot Magician — Pick-and-Place PPO Environment

## Files

| File | Purpose |
|------|---------|
| `dobot_env_cfg.py` | Scene + environment config (robots, objects, spaces, reward weights) |
| `dobot_env.py` | `DirectRLEnv` implementation (obs, reward, reset, suction gripper) |
| `dobot_kinematics.py` | **Unchanged** from previous session — IK, FK, coordinate conversion |
| `train.py` | Launch PPO training via rsl-rl |
| `record.py` | Run trained policy, save HDF5 for SFT |

## Setup

Copy all files into `C:\Users\NexusUser\thomas_sim\` alongside `dobot_kinematics.py` and `dobot.usd`.

## Training

```bat
# Headless, 64 parallel envs
isim.bat train.py --num_envs 64 --headless --max_iter 3000

# With viewer (fewer envs)
isim.bat train.py --num_envs 8 --max_iter 3000

# Resume from checkpoint
isim.bat train.py --num_envs 64 --headless --resume logs/dobot_ppo/run_001/model_1000.pt
```

Checkpoints save every 100 iterations to `logs/dobot_ppo/`.
Monitor with TensorBoard: `tensorboard --logdir logs/dobot_ppo`

## Recording SFT data

```bat
isim.bat record.py --checkpoint logs/dobot_ppo/run_001/model_3000.pt --num_episodes 200 --enable_cameras
```

Output: `dataset_hdf5/sim_ppo/episode_NNN.h5` — same format as real robot data.

## Key design decisions

### Coordinate system
Proprio is output in Dobot SDK mm space via `world_to_sdk()` — exactly matching real `GetPose()` output. This is the alignment work from the previous session, preserved here.

### Action space
4-dim Cartesian delta: `[dx, dy, dz, d_gripper]` in world metres.
- Scale: 5 mm per policy step (20 Hz effective rate, physics at 200 Hz with decimation=10)
- Gripper: continuous logit thresholded at 0 for suction on/off

### Gripper
Fake suction (same as previous session). Spring force holds block to EE when:
1. `d_gripper > 0` (gripper command on)
2. EE is within 35 mm XY and 60 mm Z of block

### Reward shaping
| Term | Weight | Signal |
|------|--------|--------|
| Reach (exp) | 1.0 | EE → block distance, only when not holding |
| Lift | 2.0 | Block height above floor, only when holding |
| Transport (exp) | 5.0 | Block → plate XY distance, when holding and lifted |
| Success | 10.0 | Block on plate, gripper released |
| Action penalty | 0.01 | L2 norm of actions (smoothness) |

### Extensibility
To add a new task (e.g. stack, sort):
1. Subclass `DobotEnvCfg` — add new objects, change zones, update `observation_space`
2. Subclass `DobotEnv` — override `_get_rewards()` and `_get_dones()`
3. The robot, kinematics, camera, and suction gripper all carry over unchanged

## Tuning guide

Start here if policy doesn't learn:
- **Gripper not triggering**: lower `action_scale_xyz` so policy can hover precisely; or widen grab thresholds in `_apply_suction`
- **Never lifts**: increase `rew_lift_weight`; check `lift_height_threshold`
- **Lifts but drops**: suction spring constant too low — increase `2000.0` in `_apply_suction`
- **Never releases**: add a small bonus for gripper=off when block is near plate
- **Too slow to converge**: increase `num_envs`; RTX A6000 should handle 256+ envs comfortably
