"""
dobot_env.py
DirectRLEnv for Dobot Magician pick-and-place.

Observation dict (returned as flat vector to rsl-rl):
  - rgb image  : (H*W*3,) float32 in [0,1]
  - proprio    : [x_mm, y_mm, z_mm, roll, pitch, yaw, gripper] in SDK space

Action space (4-dim, applied every policy step at 20 Hz):
  [dx, dy, dz]  in metres (world frame, scaled by action_scale_xyz)
  [d_grip]      continuous in [-1, 1] (thresholded to binary for grasp logic)

The environment handles:
  - IK via DobotKinematics for applying Cartesian actions
  - Fake-suction gripper using spring force on block
  - Randomised block spawn in PICK zone on reset
  - Dense reward shaping + sparse success reward
  - SFT data recording (optional, set record=True)
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, ContactSensor
from isaaclab.utils.math import sample_uniform

from dobot_env_cfg import DobotEnvCfg, PICK_X, PICK_Y, PICK_Z, PLACE_POS
from dobot_kinematics import DobotKinematics


class DobotEnv(DirectRLEnv):
    """Dobot Magician pick-and-place environment."""

    cfg: DobotEnvCfg

    # ------------------------------------------------------------------
    # Init & scene setup
    # ------------------------------------------------------------------
    def __init__(self, cfg: DobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Body / joint indices — resolved after super().__init__ spawns the robot
        self._ee_idx   = self.robot.data.body_names.index("magician_end_effector")
        self._base_idx = self.robot.data.body_names.index("magician_base_link")

        # Kinematics helper (same as before)
        self.kinematics = DobotKinematics(self.robot.data.joint_names, device=self.device)

        # Current EE position in world frame, per env  (num_envs, 3)
        self._ee_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Gripper state: True = suction on
        self._gripper_on = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Whether the block is currently held
        self._holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Previous EE pos for action delta recording
        self._prev_ee_sdk = torch.zeros(self.num_envs, 3, device=self.device)

        # Place position as tensor
        self._place_pos_w = torch.tensor(PLACE_POS[:3], device=self.device).unsqueeze(0).expand(self.num_envs, -1)

        # Joint position targets (maintained between steps for smooth motion)
        self._joint_targets = self.robot.data.default_joint_pos.clone()

        print(f"[DobotEnv] Initialised. num_envs={self.num_envs}, device={self.device}")
        print(f"[DobotEnv] EE body idx={self._ee_idx}, base body idx={self._base_idx}")

    def _setup_scene(self):
        """Spawn all assets, clone environments, register with scene."""
        # Robot
        self.robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self.robot

        # Block
        self.block = RigidObject(self.cfg.scene.block)
        self.scene.rigid_objects["block"] = self.block

        # Plate (static target)
        self.plate = RigidObject(self.cfg.scene.plate)
        self.scene.rigid_objects["plate"] = self.plate

        # Camera
        self.camera = TiledCamera(self.cfg.scene.camera)
        self.scene.sensors["camera"] = self.camera

        # Contact sensor: EE ↔ Block only (single body, filtered)
        self.ee_contact = ContactSensor(self.cfg.scene.ee_contact)
        self.scene.sensors["ee_contact"] = self.ee_contact

        # Ground plane & light (not cloned, world-level)
        sim_utils.spawn_ground_plane(prim_path="/World/ground", cfg=sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

        # Clone environments & filter collisions between env copies
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ------------------------------------------------------------------
    # Action pipeline
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        actions: (num_envs, 4) — [dx, dy, dz, d_gripper] unnormalised.
        Scale and store; joint targets are recomputed here via IK.
        """
        # Scale actions
        delta_xyz  = actions[:, :3] * self.cfg.action_scale_xyz   # metres
        grip_cmd   = actions[:, 3]                                  # raw logit

        # Update gripper state (threshold at 0)
        self._gripper_on = grip_cmd > 0.0

        # Get current EE world pos
        self._ee_pos_w = self.robot.data.body_pos_w[:, self._ee_idx, :].clone()

        # New desired EE world position
        desired_ee_w = self._ee_pos_w + delta_xyz

        # Clamp to workspace bounds (world metres)
        desired_ee_w[:, 0].clamp_(PICK_X[0] - 0.05, PLACE_POS[0] + 0.10)
        desired_ee_w[:, 1].clamp_(PICK_Y[0] - 0.05, PLACE_POS[1] + 0.10)
        desired_ee_w[:, 2].clamp_(0.0, 0.25)

        # Convert to SDK mm for IK
        desired_sdk = self.kinematics.world_to_sdk(desired_ee_w)   # (num_envs, 3)

        # IK → URDF joint targets
        sdk_angles  = self.kinematics.inverse_kinematics(desired_sdk)
        urdf_targets = self.kinematics.sdk_to_urdf_targets(sdk_angles)  # (num_envs, 5)

        self._joint_targets = urdf_targets

    def _apply_action(self):
        """Write joint targets to the articulation."""
        self.robot.set_joint_position_target(self._joint_targets)
        self._apply_suction()

    def _apply_suction(self):
        """
        Suction gripper using PhysX contact sensor.

        force_matrix_w shape: (num_envs, 1 body, 1 filter, 3)
        We read the normal contact force magnitude between EE and Block.
        If the gripper command is on AND PhysX reports contact → latch hold.
        """
        ee_w    = self.robot.data.body_pos_w[:, self._ee_idx, :]   # (N, 3)
        blk_w   = self.block.data.root_pos_w                        # (N, 3)
        blk_vel = self.block.data.root_lin_vel_w                    # (N, 3)

        # ---- Contact detection via PhysX ----
        # force_matrix_w: (N, 1, 1, 3)  — NaN when not in contact
        force_mat = self.ee_contact.data.force_matrix_w             # (N, 1, 1, 3)
        if force_mat is not None:
            contact_force_mag = force_mat[:, 0, 0, :].norm(dim=-1)  # (N,)
            in_contact = contact_force_mag > 0.1                     # Newtons threshold
        else:
            # Sensor not ready yet (first step) — no contact
            in_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ---- Grab: gripper on AND physically touching ----
        can_grab = self._gripper_on & ~self._holding & in_contact
        self._holding |= can_grab

        # ---- Release: gripper command off ----
        self._holding &= self._gripper_on

        # ---- Spring force to hold block with EE ----
        if self._holding.any():
            suction_target = ee_w.clone()
            suction_target[:, 2] -= 0.016  # air gap below EE tip

            hold_ids = self._holding.nonzero(as_tuple=False).squeeze(-1)
            err   = suction_target[hold_ids] - blk_w[hold_ids]
            vel   = blk_vel[hold_ids]
            force = 2000.0 * err - 20.0 * vel
            force[:, 2] += 0.05 * 9.81   # gravity compensation

            forces  = torch.zeros(self.num_envs, 1, 3, device=self.device)
            torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
            forces[hold_ids, 0, :] = force
            self.block.apply_external_force_torque(forces, torques)

    # ------------------------------------------------------------------
    # Done / reset
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (terminated, timed_out)."""
        # Success: block is on plate, gripper open
        blk_w   = self.block.data.root_pos_w
        plate_w = torch.tensor(PLACE_POS, device=self.device).unsqueeze(0)
        dist_to_plate = (blk_w[:, :2] - plate_w[:, :2]).norm(dim=-1)
        on_plate  = (dist_to_plate < self.cfg.success_radius) & \
                    (blk_w[:, 2] < self.cfg.success_height_max) & \
                    ~self._gripper_on

        # Failure: block fell off the workspace
        fell = blk_w[:, 2] < -0.05

        terminated = on_plate | fell
        timed_out  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids

        super()._reset_idx(env_ids)

        # ---- Reset robot ----
        root_state = self.robot.data.default_root_state[env_ids_tensor].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids_tensor]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids_tensor)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids_tensor)
        joint_pos = self.robot.data.default_joint_pos[env_ids_tensor].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids_tensor].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_tensor)
        self.robot.reset(env_ids_tensor)

        # ---- Randomise block in pick zone ----
        n = len(env_ids_tensor)
        bx = sample_uniform(PICK_X[0], PICK_X[1], (n,), device=self.device)
        by = sample_uniform(PICK_Y[0], PICK_Y[1], (n,), device=self.device)
        bz = torch.full((n,), PICK_Z, device=self.device)

        blk_state = self.block.data.default_root_state[env_ids_tensor].clone()
        blk_state[:, 0] = bx + self.scene.env_origins[env_ids_tensor, 0]
        blk_state[:, 1] = by + self.scene.env_origins[env_ids_tensor, 1]
        blk_state[:, 2] = bz
        blk_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # identity quat
        blk_state[:, 7:] = 0.0  # zero velocity
        self.block.write_root_pose_to_sim(blk_state[:, :7], env_ids=env_ids_tensor)
        self.block.write_root_velocity_to_sim(blk_state[:, 7:], env_ids=env_ids_tensor)
        self.block.reset(env_ids_tensor)

        # ---- Plate stays fixed, just reset internal buffers ----
        self.plate.reset(env_ids_tensor)

        # ---- Reset internal state ----
        self._gripper_on[env_ids_tensor] = False
        self._holding[env_ids_tensor]    = False
        self._joint_targets[env_ids_tensor] = self.robot.data.default_joint_pos[env_ids_tensor].clone()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """
        Returns dict with key "policy" → flat tensor (num_envs, obs_dim).
        obs_dim = H*W*3 + 7
        """
        # Camera: (num_envs, H, W, 3) uint8 → float [0,1]
        rgb = self.camera.data.output["rgb"][:, :, :, :3].float() / 255.0
        rgb_flat = rgb.reshape(self.num_envs, -1)   # (N, 480*640*3)

        # Proprio in SDK mm space
        proprio = self._get_proprio_sdk()            # (N, 7)

        obs = torch.cat([rgb_flat, proprio], dim=-1)
        return {"policy": obs}

    def _get_proprio_sdk(self) -> torch.Tensor:
        """
        Returns (num_envs, 7): [x_mm, y_mm, z_mm, 0, 0, 0, gripper]
        in Dobot SDK coordinate space — matches real GetPose() output.
        """
        ee_w   = self.robot.data.body_pos_w[:, self._ee_idx, :]
        base_w = self.robot.data.body_pos_w[:, self._base_idx, :]
        rel_mm = (ee_w - base_w) * 1000.0   # metres → mm

        gripper = torch.where(
            self._gripper_on,
            torch.ones(self.num_envs, device=self.device),
            -torch.ones(self.num_envs, device=self.device),
        ).unsqueeze(-1)

        zeros = torch.zeros(self.num_envs, 3, device=self.device)  # roll, pitch, yaw
        return torch.cat([rel_mm, zeros, gripper], dim=-1)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        ee_w    = self.robot.data.body_pos_w[:, self._ee_idx, :]
        blk_w   = self.block.data.root_pos_w
        plate_w = torch.tensor(PLACE_POS, device=self.device).unsqueeze(0).expand(self.num_envs, -1)

        # 1. Reach reward: EE → block (only when not holding)
        dist_ee_blk = (ee_w - blk_w).norm(dim=-1)
        reach_rew = torch.exp(-5.0 * dist_ee_blk) * (~self._holding).float()

        # 2. Lift reward: block above floor threshold
        lift_rew = torch.clamp(blk_w[:, 2] - PICK_Z, min=0.0) / self.cfg.lift_height_threshold
        lift_rew = lift_rew * self._holding.float()

        # 3. Transport reward: block → plate (only when holding and lifted)
        lifted = (blk_w[:, 2] > (PICK_Z + 0.02)) & self._holding
        dist_blk_plate = (blk_w[:, :2] - plate_w[:, :2]).norm(dim=-1)
        place_rew = torch.exp(-5.0 * dist_blk_plate) * lifted.float()

        # 4. Success: block on plate, gripper released
        dist_to_plate = (blk_w[:, :2] - plate_w[:, :2]).norm(dim=-1)
        success = (dist_to_plate < self.cfg.success_radius) & \
                  (blk_w[:, 2] < self.cfg.success_height_max) & \
                  ~self._gripper_on
        success_rew = success.float()

        # 5. Action penalty (smoothness)
        action_pen = self._actions.norm(dim=-1) * self.cfg.rew_action_penalty

        total = (
            self.cfg.rew_reach_weight   * reach_rew
            + self.cfg.rew_lift_weight  * lift_rew
            + self.cfg.rew_place_weight * place_rew
            + self.cfg.rew_success_weight * success_rew
            - action_pen
        )
        return total
