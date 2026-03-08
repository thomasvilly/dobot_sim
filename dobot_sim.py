import argparse
import random
import torch
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
simulation_app = AppLauncher(parser.parse_args()).app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg

from dobot_kinematics import DobotKinematics

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
            stiffness=50.0,
            damping=10.0,
        ),
    },
)

class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light  = AssetBaseCfg(prim_path="/World/light",  spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot  = DOBOT_CFG
    camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.5, -0.5, 0.4), rot=(0.85, 0.35, -0.15, -0.35), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955),
        width=640, height=480,
    )

def generate_episode(kinematics, robot, scene, sim, camera):
    """Executes a single pick-and-place trajectory and records the sequence."""
    
    # Matching real-world zones from record_simple.py
    Z_SAFE  = -30.0   
    Z_PICK  = -75.0  
    Z_HOVER =  50.0 
    
    pick_x  = random.uniform(100, 160)
    pick_y  = random.uniform(-120, 0)
    place_x = random.uniform(140, 220)
    place_y = random.uniform(80, 200)

    # Define the high-level PTP trajectory waypoints (X, Y, Z, Gripper)
    waypoints = [
        [150.0, 40.0, Z_SAFE, -1.0],      # 1. Start at safe home
        [pick_x, pick_y, Z_HOVER, -1.0],  # 2. Hover over pick
        [pick_x, pick_y, Z_PICK,   1.0],  # 3. Drop down & GRAB
        [pick_x, pick_y, Z_HOVER,  1.0],  # 4. Lift up
        [place_x, place_y, Z_HOVER, 1.0], # 5. Move to place hover
        [place_x, place_y, Z_PICK, -1.0], # 6. Drop down & RELEASE
        [place_x, place_y, Z_HOVER, -1.0] # 7. Lift up
    ]

    episode_data = []
    current_pos = torch.tensor(waypoints[0][:3], device="cuda").unsqueeze(0)
    
    print(f"[INFO] Starting Episode: Pick({pick_x:.1f}, {pick_y:.1f}) -> Place({place_x:.1f}, {place_y:.1f})")

    for target in waypoints:
        target_pos = torch.tensor(target[:3], device="cuda").unsqueeze(0)
        gripper_state = torch.tensor([[target[3]]], device="cuda")
        
        # Calculate how many 20Hz steps to take based on distance (speed control)
        dist = torch.norm(target_pos - current_pos).item()
        steps = max(10, int(dist / 2.0)) # Approx 2mm per frame

        for step in range(steps):
            # Interpolate to create a smooth path
            alpha = step / float(steps)
            interp_pos = current_pos + alpha * (target_pos - current_pos)
            
            # 1. Math: XYZ -> SDK Angles -> URDF Targets
            sdk_angles = kinematics.inverse_kinematics(interp_pos)
            urdf_targets = kinematics.sdk_to_urdf_targets(sdk_angles)
            
            # 2. Physics: Apply targets and step sim
            robot.set_joint_position_target(urdf_targets)
            
            # Step physics 10 times (200Hz) to simulate 1 control frame (20Hz)
            for _ in range(10):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())

            # 3. Record: Capture State
            proprio = kinematics.get_proprio(robot.data.joint_pos, gripper_state)
            
            # Assuming camera is enabled and returning valid data
            img = camera.data.output["rgb"][0].cpu().numpy() if "rgb" in camera.data.output else None
            
            episode_data.append({
                "action": [*interp_pos[0].cpu().numpy(), gripper_state.item()],
                "proprio": proprio[0].cpu().numpy(),
                "image": img
            })
            
        current_pos = target_pos # Update for next waypoint

    return episode_data

def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, gravity=(0.0, 0.0, -9.81)))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot  = scene["robot"]
    camera = scene["camera"]
    kinematics = DobotKinematics(robot.data.joint_names, device="cuda")

    # Settle the physics engine
    print("[INFO] Settling...")
    for _ in range(400):
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    print("[INFO] Settled. Generating episodes...")

    # Run 3 test episodes
    for i in range(3):
        ep_data = generate_episode(kinematics, robot, scene, sim, camera)
        print(f"--> Episode {i+1} completed with {len(ep_data)} frames.")

    print("\n[INFO] Script complete.")

if __name__ == "__main__":
    main()
    simulation_app.close()