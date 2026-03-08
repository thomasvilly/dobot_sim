import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
simulation_app = AppLauncher(parser.parse_args()).app

import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
import math

DOBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/NexusUser/thomas_sim/dobot.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.212),
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
            stiffness=100.0,
            damping=10.0,
        ),
    },
)

class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light  = AssetBaseCfg(prim_path="/World/light",  spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot  = DOBOT_CFG
    # Stationary camera looking at the robot from the front-side, matching real setup
    camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, -0.5, 0.4),   # 50cm front, 50cm to side, 40cm up
            rot=(0.85, 0.35, -0.15, -0.35),  # angled down toward robot
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
        width=640,
        height=480,
    )

def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, gravity=(0.0, 0.0, -9.81)))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot      = scene["robot"]
    camera     = scene["camera"]
    joint_names = robot.data.joint_names

    def get_sim_proprio(gripper_on=False):
        # 1. Read actual simulated joint positions
        joints = robot.data.joint_pos[0].cpu().numpy()
        j1 = joints[joint_names.index("magician_joint_1")]
        j2 = joints[joint_names.index("magician_joint_2")]
        j3_rel = joints[joint_names.index("magician_joint_3")]
        
        # 2. Reconstruct the absolute SDK J3 angle
        j3 = j3_rel + j2
        
        # 3. Dobot SDK Forward Kinematics (Forearm Tip)
        radius = 135.0 * math.sin(j2) + 147.0 * math.cos(j3)
        z = 135.0 * math.cos(j2) - 147.0 * math.sin(j3)
        
        x = radius * math.cos(j1)
        y = radius * math.sin(j1)
        
        grip = 1.0 if gripper_on else -1.0
        return np.array([x, y, z, grip], dtype=np.float32)

    hold = robot.data.default_joint_pos.clone()
    j2   = joint_names.index("magician_joint_2")
    j3   = joint_names.index("magician_joint_3")
    jp   = joint_names.index("magician_joint_end_pitch")
    j1   = joint_names.index("magician_joint_1")
    j4_idx = joint_names.index("magician_joint_4")

    # Settle
    print("[INFO] Settling...")
    for i in range(400):
        robot.set_joint_position_target(hold)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    print("[INFO] Settled. Running...")
    proprio = get_sim_proprio()
    print(f"  Initial proprio (mm): x={proprio[0]:.1f}  y={proprio[1]:.1f}  z={proprio[2]:.1f}  grip={proprio[3]}")

    # Verify camera frame shape
    rgb = camera.data.output["rgb"][0].cpu().numpy()
    print(f"  Camera frame shape: {rgb.shape}  dtype: {rgb.dtype}  min/max: {rgb.min()}/{rgb.max()}")

    # Three real-world calibration poses (joint angles in DEGREES, converted to radians)
    cal_poses = {
        "down":  {"j1": -50.74, "j2": 62.47, "j3": 84.07, "j4": -26.57},
        "home":  {"j1":  26.55, "j2": 39.98, "j3": 21.34, "j4": -26.57},
        "up":    {"j1":  86.35, "j2": 38.34, "j3": -5.52, "j4": -26.57},
    }

    real_xyz = {
        "down": (85.4,  -104.5,  -83.8),
        "home": (200.1,  100.0,   50.0),
        "up":   (14.7,   229.6,  120.0),
    }

    for name, joints in cal_poses.items():
        target = torch.zeros(1, len(joint_names), device="cuda")
        
        # Base rotation remains 1:1
        target[:, j1] = math.radians(joints["j1"])
        
        # J2 in SDK is angle from vertical. URDF 0 is vertical. (1:1 mapping)
        target[:, j2] = math.radians(joints["j2"])
        
        # J3 in SDK is the absolute angle of Link 3 below horizontal. 
        # URDF requires the relative angle from Link 2.
        target[:, j3] = math.radians(joints["j3"] - joints["j2"])
        
        # Map SDK j4 (yaw) to the actual yaw joint
        target[:, j4_idx] = math.radians(joints["j4"])
        
        # Visually level the suction cup (Pitch opposes total Link 3 angle)
        target[:, jp] = math.radians(-joints["j3"])

        # Settle at this pose
        for _ in range(100):
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        sim_p = get_sim_proprio()
        rx, ry, rz = real_xyz[name]
        print(f"\n[{name}]")
        print(f"  Real: x={rx:.1f}  y={ry:.1f}  z={rz:.1f}")
        print(f"  Sim:  x={sim_p[0]:.1f}  y={sim_p[1]:.1f}  z={sim_p[2]:.1f}")
        print(f"  Diff: x={sim_p[0]-rx:.1f}  y={sim_p[1]-ry:.1f}  z={sim_p[2]-rz:.1f}")

if __name__ == "__main__":
    main()
    simulation_app.close()

# isim.bat C:\Users\NexusUser\IsaacLab\scripts\tools\convert_urdf.py C:\Users\NexusUser\thomas_sim\dobot.urdf C:\Users\NexusUser\thomas_sim\dobot.usd --fix-base --joint-target-type none