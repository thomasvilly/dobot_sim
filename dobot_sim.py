import argparse
import random
import os
import h5py
import torch
import math
import numpy as np
import glob
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
simulation_app = AppLauncher(parser.parse_args()).app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg

from dobot_kinematics import DobotKinematics

def euler_to_quat(roll, pitch, yaw):
    cr = math.cos(math.radians(roll) * 0.5)
    sr = math.sin(math.radians(roll) * 0.5)
    cp = math.cos(math.radians(pitch) * 0.5)
    sp = math.sin(math.radians(pitch) * 0.5)
    cy = math.cos(math.radians(yaw) * 0.5)
    sy = math.sin(math.radians(yaw) * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)

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
            stiffness=400.0, 
            damping=40.0,
        ),
    },
)

class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light  = AssetBaseCfg(prim_path="/World/light",  spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot  = DOBOT_CFG
    
    block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)), 
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, -0.05, 0.015)),
    )
    
    camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.60, 0.00, 0.30), # Moved 20cm back, 5cm up
            rot=euler_to_quat(roll=0, pitch=45, yaw=180), # Pitched slightly more forward
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955),
        width=640, height=480,
    )

def generate_episode(kinematics, robot, scene, sim, camera):
    block = scene["block"]
    
    Z_SAFE  = -30.0   
    Z_PICK  = -75.0  
    Z_HOVER =  50.0 
    
    pick_x  = random.uniform(100, 160)
    pick_y  = random.uniform(-120, 0)
    place_x = random.uniform(140, 220)
    place_y = random.uniform(80, 200)

    block_state = block.data.default_root_state.clone()
    block_state[:, 0] = pick_x / 1000.0
    block_state[:, 1] = pick_y / 1000.0
    block_state[:, 2] = 0.015 
    block.write_root_state_to_sim(block_state)

    waypoints = [
        [150.0, 40.0, Z_SAFE, -1.0],      
        [pick_x, pick_y, Z_HOVER, -1.0],  
        [pick_x, pick_y, Z_PICK,   1.0],  
        [pick_x, pick_y, Z_HOVER,  1.0],  
        [place_x, place_y, Z_HOVER, 1.0], 
        [place_x, place_y, Z_PICK, -1.0], 
        [place_x, place_y, Z_HOVER, -1.0] 
    ]

    episode_data = []
    current_pos = torch.tensor(waypoints[0][:3], device="cuda").unsqueeze(0)
    is_holding = False
    
    print(f"[INFO] Generating Episode: Pick({pick_x:.1f}, {pick_y:.1f}) -> Place({place_x:.1f}, {place_y:.1f})")

    SPEED_MM_PER_FRAME = 20.0 

    for target in waypoints:
        target_pos = torch.tensor(target[:3], device="cuda").unsqueeze(0)
        gripper_state = torch.tensor([[target[3]]], device="cuda")
        
        dist = torch.norm(target_pos - current_pos).item()
        steps = max(2, int(dist / SPEED_MM_PER_FRAME))

        for step in range(steps):
            alpha = step / float(steps)
            interp_pos = current_pos + alpha * (target_pos - current_pos)
            
            sdk_angles = kinematics.inverse_kinematics(interp_pos)
            urdf_targets = kinematics.sdk_to_urdf_targets(sdk_angles)
            robot.set_joint_position_target(urdf_targets)
            
            for _ in range(6):
                actual_proprio = kinematics.get_proprio(robot.data.joint_pos, gripper_state)
                actual_tcp = actual_proprio[0, :3] / 1000.0
                actual_tcp[2] += 0.112 

                forces = torch.zeros((1, 1, 3), device="cuda")
                torques = torch.zeros((1, 1, 3), device="cuda")
                
                if gripper_state.item() == 1.0:
                    block_pos = block.data.root_pos_w[0]
                    
                    if not is_holding:
                        dist_xy = torch.norm(actual_tcp[:2] - block_pos[:2]).item()
                        dist_z = abs((actual_tcp[2] - 0.015) - block_pos[2]).item()
                        if dist_xy < 0.03 and dist_z < 0.03: # Tightened grab threshold
                            is_holding = True
                    
                    if is_holding:
                        block_vel = block.data.root_lin_vel_w[0]
                        suction_target = actual_tcp.clone()
                        
                        # 0.015 (block half-height) + 0.001 (air gap to stop collisions)
                        suction_target[2] -= 0.016 
                        
                        # Unclamped, ultra-stiff, mathematically perfectly damped spring
                        kp = 2000.0  
                        kd = 20.0    # kd = 2 * sqrt(kp * mass) = 2 * sqrt(2000 * 0.05)
                        
                        force = (kp * (suction_target - block_pos)) - (kd * block_vel)
                        force[2] += 0.05 * 9.81 
                        
                        forces[0, 0] = force
                else:
                    is_holding = False
                    
                block.set_external_force_and_torque(forces, torques)
                
                scene.write_data_to_sim()
                sim.step()

            scene.update(sim.get_physics_dt() * 6)
            proprio = kinematics.get_proprio(robot.data.joint_pos, gripper_state)
            
            if "rgb" in camera.data.output:
                img = camera.data.output["rgb"][0].cpu().numpy()
                if img.shape[-1] == 4: img = img[:, :, :3] 
                
                state_arr = proprio[0].cpu().numpy()
                episode_data.append({
                    "action": state_arr.copy(),
                    "state": state_arr.copy(),
                    "top": img
                })
            
        current_pos = target_pos

    return episode_data

def save_hdf5(buffer, dataset_dir="dataset_hdf5/sim_session"):
    if not buffer: return
    os.makedirs(dataset_dir, exist_ok=True)
    
    next_id = len(glob.glob(os.path.join(dataset_dir, "episode_*.h5"))) + 1
    fname = os.path.join(dataset_dir, f"episode_{next_id:03d}.h5")
    
    print(f"--> Saving {fname}...")
    with h5py.File(fname, 'w') as f:
        f.create_dataset('observations/images/top', data=np.array([b['top'] for b in buffer]), compression="gzip")
        f.create_dataset('observations/state', data=np.array([b['state'] for b in buffer]))
        f.create_dataset('action', data=np.array([b['action'] for b in buffer]))
        f.attrs['instruction'] = "pick up the blue block and put it on the plate"
    print("Saved.")

def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, gravity=(0.0, 0.0, -9.81)))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot  = scene["robot"]
    camera = scene["camera"]
    kinematics = DobotKinematics(robot.data.joint_names, device="cuda")

    print("[INFO] Settling...")
    for _ in range(200):
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    ep_data = generate_episode(kinematics, robot, scene, sim, camera)
    print(f"--> Episode completed with {len(ep_data)} frames.")
    
    save_hdf5(ep_data)

if __name__ == "__main__":
    main()
    simulation_app.close()