"""
SMPL (Skinned Multi-Person Linear Model) implementation in PyTorch.
This module provides functionality for 3D human body modeling using the SMPL model.
"""

import os
from pathlib import Path
import pickle
from typing import Optional, Tuple, Dict, Union

import numpy as np
import torch
from torch.nn import Module, Parameter

def rodrigues(r: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' rotation formula that converts axis-angle to rotation matrix.

    Args:
        r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3]

    Returns:
        Rotation matrix of shape [batch_size * angle_num, 3, 3]
    """
    eps = torch.randn_like(r) * 1e-8
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
    theta_dim = theta.shape[0]
    r_hat = r / theta

    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=r.dtype, device=r.device)

    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
        r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
        -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1
    ).reshape(-1, 3, 3)

    i_cube = torch.eye(3, dtype=r.dtype, device=r.device).unsqueeze(0).expand(theta_dim, -1, -1)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)

    return cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m

def with_zeros(x: torch.Tensor) -> torch.Tensor:
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

    Args:
        x: Input tensor of shape [batch_size, 3, 4]

    Returns:
        Tensor of shape [batch_size, 4, 4] with [0, 0, 0, 1] appended
    """
    ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=x.dtype
    ).expand(x.shape[0], -1, -1).to(x.device)
    return torch.cat((x, ones), dim=1)

def pack(x: torch.Tensor) -> torch.Tensor:
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

    Args:
        x: Input tensor of shape [batch_size, 4, 1]

    Returns:
        Tensor of shape [batch_size, 4, 4] with zeros appended
    """
    zeros43 = torch.zeros(
        (x.shape[0], x.shape[1], 4, 3), dtype=x.dtype).to(x.device)
    return torch.cat((zeros43, x), dim=3)

def write_obj(vertices: torch.Tensor, faces: torch.Tensor, filename: str) -> None:
    """
    Write mesh data to Wavefront OBJ file format.

    Args:
        vertices: Vertex coordinates of shape [V, 3]
        faces: Face indices of shape [F, 3]
        filename: Output file path
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for f in faces + 1:  # Add 1 since OBJ uses 1-based indexing
            fp.write(f'f {f[0]} {f[1]} {f[2]}\n')

class SMPLModel(Module):
    """
    SMPL model implementation that handles body shape and pose parameters to generate
    3D mesh vertices and joint positions.
    """

    def __init__(self,
                 model_path: str = './SMPL_NEUTRAL.pkl',
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the SMPL model.

        Args:
            model_path: Path to the SMPL model file
            dtype: Data type for model parameters
            device: Device to store the model parameters

        Raises:
            FileNotFoundError: If model_path does not exist
            ValueError: If model file is corrupted or invalid
        """
        super().__init__()

        self.dtype = dtype

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SMPL model not found at: {model_path}")

        try:
            with open(model_path, 'rb') as f:
                params = pickle.load(f, encoding='latin')
        except Exception as e:
            raise ValueError(f"Failed to load SMPL model: {str(e)}")

        # Load joint regressors
        J_regressor = torch.from_numpy(np.array(params['J_regressor'].todense())).type(dtype)
        lsp_regressor_path = Path(__file__).parent / 'smpl/J_regressor_lsp.npy'
        joint_regressor = torch.from_numpy(np.load(str(lsp_regressor_path))).type(dtype)

        # Load SMPL model parameters
        weights = torch.from_numpy(params['weights']).type(dtype)
        posedirs = torch.from_numpy(params['posedirs']).type(dtype)
        v_template = torch.from_numpy(params['v_template']).type(dtype)
        shapedirs = torch.from_numpy(params['shapedirs']).type(dtype)
        faces = torch.from_numpy(params['f'].astype(np.int32)).type(torch.long)
        kintree_table = params['kintree_table']

        # Register model parameters as buffers
        for name, tensor in [
            ('J_regressor', J_regressor),
            ('joint_regressor', joint_regressor),
            ('weights', weights),
            ('posedirs', posedirs),
            ('v_template', v_template),
            ('shapedirs', shapedirs),
            ('faces', faces),
        ]:
            self.register_buffer(name, tensor)

        self.kintree_table = kintree_table

        # Create parent joint mapping
        id_to_col = {
            self.kintree_table[1, i]: i
            for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }


    def _compute_global_transforms(self, lRs: torch.Tensor, J: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute global transformation matrices for the SMPL model.

        This is an internal method that implements the SMPL model's global transformation computation.
        It transforms joints from local to global coordinates while handling the kinematic tree structure.

        Args:
            lRs: Local rotation matrices [batch_size, 24, 3, 3]
            J: Joint locations [batch_size, 24, 3]
            scale: Scale factor [batch_size, 1]

        Returns:
            Tuple containing:
                - Global transformation matrices [batch_size, 24, 4, 4]
                - Scaled rotation matrices [batch_size, 24, 3, 3]
        """
        batch_num = lRs.shape[0]
        lRs = lRs.clone()
        lRs[:, 0] *= scale.reshape(batch_num, 1, 1)
        results = []

        # Handle root joint
        results.append(
            with_zeros(torch.cat((lRs[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
        )

        # Process kinematic chain
        for i in range(1, self.kintree_table.shape[1]):
            parent_transform = results[self.parent[i]]
            local_transform = with_zeros(
                torch.cat(
                    (lRs[:, i], torch.reshape(J[:, i, :] - J[:, self.parent[i], :], (-1, 3, 1))),
                    dim=2
                )
            )
            results.append(torch.matmul(parent_transform, local_transform))

        # Stack and compute final transformations
        stacked = torch.stack(results, dim=1)
        joints_homogeneous = torch.cat(
            (J, torch.zeros((batch_num, 24, 1), dtype=self.dtype, device=lRs.device)),
            dim=2
        ).reshape(batch_num, 24, 4, 1)

        deformed_joints = torch.matmul(stacked, joints_homogeneous)
        results = stacked - pack(deformed_joints)

        return results, lRs


    def forward(
            self,
            betas: Optional[torch.Tensor] = None,
            thetas: Optional[torch.Tensor] = None,
            trans: Optional[torch.Tensor] = None,
            scale: Optional[torch.Tensor] = None,
            lsp: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate 3D mesh vertices and joint positions from SMPL parameters.

        Args:
            betas: Shape parameters [batch_size, 10]
            thetas: Pose parameters [batch_size, 24, 3] for axis-angle or [batch_size, 24, 3, 3] for rotation matrices
            trans: Translation parameters [batch_size, 3]
            scale: Scale parameters [batch_size, 1]
            lsp: Whether to use LSP joint regressor instead of default

        Returns:
            Tuple containing:
                - Mesh vertices [batch_size, 6890, 3]
                - Joint positions [batch_size, 24, 3] or [batch_size, 14, 3] if lsp=True

        Raises:
            ValueError: If input parameters have invalid shapes or values
        """
        device = self.weights.device

        # Input validation
        if betas is None or thetas is None:
            raise ValueError("Both betas and thetas must be provided")

        if betas.dim() != 2 or betas.size(1) != 10:
            raise ValueError(f"Expected betas shape [batch_size, 10], got {list(betas.shape)}")

        thetas_shape = thetas.shape[1:]
        if thetas_shape not in [(24, 3), (24, 3, 3)]:
            raise ValueError(
                f"Expected thetas shape [batch_size, 24, 3] or [batch_size, 24, 3, 3], got {list(thetas.shape)}"
            )

        batch_size = betas.shape[0]
        theta_is_matrix = thetas_shape == (24, 3, 3)
        current_scale = torch.tensor([1,]*batch_size, device=device) if scale is None else scale

        # Compute base shape from shape parameters
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)

        # Convert pose parameters to rotation matrices if needed
        if not theta_is_matrix:
            thetas = rodrigues(thetas.view(-1, 1, 3)).reshape(batch_size, -1, 3, 3)
        else:
            thetas = thetas.view(batch_size, -1, 3, 3)

        # Compute global transformations
        G, R_cube_big = self._compute_global_transforms(thetas, J, current_scale)

        # Pose shape blending
        R_cube = R_cube_big[:, 1:, :, :]
        I_cube = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(0).expand(batch_size, R_cube.shape[1], -1, -1)
        lrotmin = (R_cube - I_cube).reshape(batch_size, -1)
        v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))

        # Linear blend skinning
        T = torch.tensordot(G, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)
        rest_shape_h = torch.cat(
            (v_posed, torch.ones((batch_size, v_posed.shape[1], 1), dtype=self.dtype, device=device)),
            dim=2
        )
        v = torch.matmul(T, rest_shape_h.reshape(batch_size, -1, 4, 1))
        vertices = v.reshape(batch_size, -1, 4)[:, :, :3]

        # Apply translation if provided
        if trans is not None:
            vertices = vertices + trans.reshape(batch_size, 1, 3)

        # Compute joint locations
        regressor = self.joint_regressor if lsp else self.J_regressor
        joints = torch.tensordot(vertices, regressor.transpose(0, 1), dims=([1], [0])).transpose(1, 2)

        return vertices, joints
