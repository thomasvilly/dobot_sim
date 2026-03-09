import torch
import math

class DobotKinematics:
    def __init__(self, joint_names, device="cuda"):
        self.device = device
        
        # Link lengths based on Dobot physical specs
        self.L2 = 135.0
        self.L3 = 147.0

        # Store joint indices for fast tensor slicing
        self.j1_idx = joint_names.index("magician_joint_1")
        self.j2_idx = joint_names.index("magician_joint_2")
        self.j3_idx = joint_names.index("magician_joint_3")
        self.j4_idx = joint_names.index("magician_joint_4")
        self.jp_idx = joint_names.index("magician_joint_end_pitch")
        
        self.num_joints = len(joint_names)

    def inverse_kinematics(self, target_pos):
        """
        Calculates SDK joint angles from Cartesian coordinates (X, Y, Z).
        target_pos: Tensor of shape (num_envs, 3) -> [X, Y, Z] in mm.
        """
        x = target_pos[:, 0]
        y = target_pos[:, 1]
        z = target_pos[:, 2]

        j1 = torch.atan2(y, x)
        r = torch.sqrt(x**2 + y**2)
        d = torch.sqrt(r**2 + z**2)
        gamma = torch.atan2(r, z)

        cos_alpha = (self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d)
        cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
        alpha = torch.acos(cos_alpha)

        j2 = gamma - alpha

        cos_j3 = (r - self.L2 * torch.sin(j2)) / self.L3
        sin_j3 = (self.L2 * torch.cos(j2) - z) / self.L3
        j3 = torch.atan2(sin_j3, cos_j3)
        
        j4 = torch.full_like(j1, math.radians(-26.57))

        return torch.stack([j1, j2, j3, j4], dim=-1)

    def sdk_to_urdf_targets(self, sdk_targets):
        """Maps SDK angles to URDF constraints (Relative J3 & Parallelogram)."""
        num_envs = sdk_targets.shape[0]
        urdf_targets = torch.zeros((num_envs, self.num_joints), device=self.device)

        urdf_targets[:, self.j1_idx] = sdk_targets[:, 0]
        urdf_targets[:, self.j2_idx] = sdk_targets[:, 1]
        urdf_targets[:, self.j3_idx] = sdk_targets[:, 2] - sdk_targets[:, 1]
        urdf_targets[:, self.jp_idx] = -sdk_targets[:, 2]
        urdf_targets[:, self.j4_idx] = sdk_targets[:, 3]

        return urdf_targets

    def get_proprio(self, current_joint_pos, gripper_states):
        j1 = current_joint_pos[:, self.j1_idx]
        j2 = current_joint_pos[:, self.j2_idx]
        j3_rel = current_joint_pos[:, self.j3_idx]

        j3_abs = j3_rel + j2

        radius = self.L2 * torch.sin(j2) + self.L3 * torch.cos(j3_abs)
        z = self.L2 * torch.cos(j2) - self.L3 * torch.sin(j3_abs)
        x = radius * torch.cos(j1)
        y = radius * torch.sin(j1)

        batch_size = x.shape[0]
        zeros = torch.zeros(batch_size, device=self.device)

        # Returns: [X, Y, Z, R (0), J1 (0), J2 (0), Grip]
        return torch.stack([x, y, z, zeros, zeros, zeros, gripper_states.squeeze(-1)], dim=-1)