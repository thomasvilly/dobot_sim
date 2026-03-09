import argparse
import random
import os
import h5py
import torch
import numpy as np
import glob
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
            stiffness=150.0, # Kept high for fast 30Hz tracking
            damping=15.0,
        ),
    },
)

class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light  = AssetBaseCfg(prim_path="/World/light",  spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot  = DOBOT_CFG
    camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.45, 0.10, 0.35), # 45cm forward, 10cm right, 35cm up
            rot=(0.176, 0.380, 0.825, -0.380), # Angled back at robot
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955),
        width=640, height=480,
    )

def generate_episode(kinematics, robot, scene, sim, camera):
    Z_SAFE  = -30.0   
    Z_PICK  = -75.0  
    Z_HOVER =  50.0 
    
    pick_x  = random.uniform(100, 160)
    pick_y  = random.uniform(-120, 0)
    place_x = random.uniform(140, 220)
    place_y = random.uniform(80, 200)

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
    
    print(f"[INFO] Generating Episode: Pick({pick_x:.1f}, {pick_y:.1f}) -> Place({place_x:.1f}, {place_y:.1f})")

    SPEED_MM_PER_FRAME = 8.0 # 8mm * 30Hz = 240mm/sec (matches your 4-second real-world speed)

    for target in waypoints:
        target_pos = torch.tensor(target[:3], device="cuda").unsqueeze(0)
        gripper_state = torch.tensor([[target[3]]], device="cuda")
        
        dist = torch.norm(target_pos - current_pos).item()
        steps = max(3, int(dist / SPEED_MM_PER_FRAME))

        for step in range(steps):
            alpha = step / float(steps)
            interp_pos = current_pos + alpha * (target_pos - current_pos)
            
            sdk_angles = kinematics.inverse_kinematics(interp_pos)
            urdf_targets = kinematics.sdk_to_urdf_targets(sdk_angles)
            robot.set_joint_position_target(urdf_targets)
            
            # 30Hz Control: 200Hz physics dt=0.005. Stepping 6 times = 0.03s per control frame
            for _ in range(6):
                scene.write_data_to_sim()
                sim.step()

            # Render camera once per control frame
            scene.update(sim.get_physics_dt() * 6)

            proprio = kinematics.get_proprio(robot.data.joint_pos, gripper_state)
            
            if "rgb" in camera.data.output:
                img = camera.data.output["rgb"][0].cpu().numpy()
                if img.shape[-1] == 4: img = img[:, :, :3] # RGBA to RGB
                
                state_arr = proprio[0].cpu().numpy()
                episode_data.append({
                    "action": state_arr.copy(), # Pre-conversion format stores absolute states in action
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

    # Generate 1 Test Episode
    ep_data = generate_episode(kinematics, robot, scene, sim, camera)
    print(f"--> Episode completed with {len(ep_data)} frames.")
    
    save_hdf5(ep_data)

if __name__ == "__main__":
    main()
    simulation_app.close()