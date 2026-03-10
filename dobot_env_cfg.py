"""
dobot_env_cfg.py
Configuration for the Dobot Magician pick-and-place DirectRLEnv.
"""
from __future__ import annotations

import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


def euler_to_quat(roll_deg, pitch_deg, yaw_deg):
    """Convert Euler angles (degrees) to quaternion (w, x, y, z)."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
DOBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/NexusUser/thomas_sim/dobot.usd",
        activate_contact_sensors=True,  # required for ContactSensor to work
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.112),
        joint_pos={
            "magician_joint_1":         0.0,
            "magician_joint_2":         0.8,
            "magician_joint_3":         0.0,
            "magician_joint_end_pitch": -0.8,
            "magician_joint_4":         0.0,
        },
    ),
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=["magician_joint_.*"],
            stiffness=400.0,
            damping=40.0,
        ),
    },
)

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class DobotSceneCfg(InteractiveSceneCfg):
    """Scene: robot + block + plate + camera."""

    # NOTE: ground plane and lights are spawned in DobotEnv._setup_scene(),
    # not here — InteractiveScene only accepts articulations/rigid objects/sensors.

    robot: ArticulationCfg = DOBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Block to pick up — spawned at default pos; _reset_idx randomises it.
    block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Block",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, -0.05, 0.015)),
    )

    # Contact sensor on the end effector, filtered to only report contact with the block.
    # Single body prim path required — force_matrix_w returns None for multi-body sensors.
    ee_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/magician_end_effector",
        update_period=0.0,       # update every physics step
        history_length=1,
        debug_vis=False,         # set True to see contact markers in viewport
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Block"],
    )

    # Plate / target — static cylinder, just visual reference.
    plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.CylinderCfg(
            radius=0.06,
            height=0.005,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.14, 0.0025)),
    )

    # Camera is intentionally omitted here.
    # It is added dynamically in _setup_scene() only when use_camera=True
    # (i.e. during record.py, not during train.py).
    camera: TiledCameraCfg | None = None


# ---------------------------------------------------------------------------
# Workspace constants (world-frame metres, matching real SDK after world_to_sdk)
# ---------------------------------------------------------------------------
# Pick zone in world metres (x, y).  z is floor level + half block height.
PICK_X = (0.100, 0.160)   # metres
PICK_Y = (-0.120, 0.000)  # metres
PICK_Z = 0.015            # metres  (floor + 0.5*block_height)

# Place / plate centre in world metres
PLACE_POS = (0.18, 0.14, 0.0025)  # plate centre (same as plate init_state)
PLACE_RADIUS = 0.06               # success if block within this radius of plate centre


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------
@configclass
class DobotEnvCfg(DirectRLEnvCfg):
    # Set True only in record.py to enable TiledCamera
    use_camera: bool = False

    # ---- timing ----
    # physics at 200 Hz, policy at 20 Hz  →  decimation = 10
    decimation: int = 10
    episode_length_s: float = 15.0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=decimation)

    scene: DobotSceneCfg = DobotSceneCfg(num_envs=64, env_spacing=2.5)

    # ---- spaces ----
    # Actions: [dx, dy, dz, d_gripper]  (Cartesian deltas in metres + gripper)
    action_space: int = 4
    # Observations for training: compact proprio only (14-dim)
    # ee_pos_mm(3) + block_pos_mm(3) + block_to_plate_mm(3) + gripper(1) +
    # ee_vel_mm(3) + holding(1)  = 14
    # Camera RGB is only used in record.py.
    observation_space: int = 14
    state_space: int = 0

    # ---- task-specific ----
    # Action scale: max delta per step (metres / gripper units)
    action_scale_xyz: float = 0.005    # 5 mm per policy step at 20 Hz
    action_scale_grip: float = 1.0

    # Reward weights — easy to tune from outside
    rew_reach_weight: float = 1.0
    rew_lift_weight: float  = 2.0
    rew_place_weight: float = 5.0
    rew_success_weight: float = 10.0
    rew_action_penalty: float = 0.01

    # Thresholds
    lift_height_threshold: float = 0.05    # metres above floor
    success_radius: float = PLACE_RADIUS
    success_height_max: float = 0.04       # block must be low on the plate
