
"""
Rotation Representation Transforms

For 'torch.tensor' data, can use 'pytorch3d.transforms' package.

Author: GeYuting

"""

import numpy as np
from typing import Tuple
import torch
import torch.nn.functional as F

# note: 单个旋转矩阵转旋转向量，或者旋转向量转旋转矩阵，都可以使用 cv2.Rodrigues 实现

## ===================
def batch_rotVec2Mat_ndarray(rotvec: np.ndarray) -> np.ndarray:
    '''
    Calculates the rotation matrices for a batch of rotation vectors.

    Parameter
    ----
    rotvec : ndarray
        `[N, 3]`, array of N axis-angle vectors

    Return
    ----
    R : ndarray
        `[N, 3, 3]`, The rotation matrices for the given axis-angle parameters

    '''

    batch_size = rotvec.shape[0]
    angle = np.linalg.norm(rotvec + 1e-8, axis=1, keepdims=True)
    small_angles = angle < 1e-6
    rot_dir = np.where(small_angles, rotvec, rotvec / angle)

    cos = np.expand_dims(np.cos(angle), axis=1)
    sin = np.expand_dims(np.sin(angle), axis=1)

    # B x 1 arrays
    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros((batch_size, 1), dtype=rotvec.dtype)
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1).reshape((batch_size, 3, 3))

    ident = np.expand_dims(np.eye(3, dtype=rotvec.dtype), axis=0)
    rotmat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rotmat




def batch_rotVec2Mat_tensor(rotvec: torch.Tensor) -> torch.Tensor:
    '''
    Calculates the rotation matrices for a batch of rotation vectors.

    Parameter
    ----
    rotvec : Tensor
        `[N, 3]`, tensor of N axis-angle vectors

    Return
    ----
    R : Tensor
        `[N, 3, 3]`, The rotation matrices for the given axis-angle parameters

    '''

    batch_size = rotvec.shape[0]
    angle = torch.linalg.norm(rotvec + 1e-8, dim=1, keepdim=True)

    small_angles = angle < 1e-6
    rot_dir = torch.where(small_angles, rotvec, rotvec / angle) # fixme: may lost the gradiant

    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    # B x 1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=rotvec.dtype, device=rotvec.device)
    K = torch.stack([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).reshape((batch_size, 3, 3))

    ident = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).unsqueeze(0).expand(batch_size, -1, -1)
    rotmat = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rotmat




## -------------------------
def batch_rotMat2Vec_ndarray(rotmat:np.ndarray, strict_mode:bool=True) -> np.ndarray:
    '''
    Calculates the rotation vectors for a batch of rotation matrices.

    Parameter
    ----
    rotmat : ndarray
        `[N, 3, 3]`, The rotation matrices for the given axis-angle parameters
    strict_mode : bool
        if `True`, raise error when inputs matrix don't pass the orthognal check or det check, else will produce an approximate matrix using SVD.

    Returns
    ----
    rot_vectors : np.ndarray
        `[N, 3]`, array of N axis-angle vectors

    '''
    _eps = 1e-5

    R_shape = rotmat.shape
    batch_size = R_shape[0]

    if R_shape[1:] != (3, 3):
        raise ValueError("Input matrix should be of shape (n, 3, 3).")

    ortho_check = np.allclose(rotmat @ rotmat.transpose(0, 2, 1),
                              np.eye(3, dtype=rotmat.dtype), rtol=_eps, atol=_eps)
    det_check = np.allclose(np.linalg.det(rotmat), 1., rtol=_eps, atol=_eps)

    if (not det_check) or (not ortho_check):
        if strict_mode:
            raise ValueError("Input rotation matrices' Determinantis is not 1. or not Orthogonal.")
        else:
            u, s, vh = np.linalg.svd(rotmat) # produce an approximate standard rotate matrix using SVD.
            rotmat = u @ vh

    trace = np.trace(rotmat, axis1=1, axis2=2)
    trace = np.clip(trace, -1., 3.) # make sure that cos_theta values in [-1, 1].

    theta = np.arccos((trace - 1.) / 2.)
    zero_mask = np.abs(theta) < _eps
    v = np.zeros((batch_size, 3), dtype=rotmat.dtype)

    R_non_zero = rotmat[~zero_mask]
    theta_non_zero = theta[~zero_mask]
    sin_theta_non_zero = np.sin(theta_non_zero) + _eps

    v[~zero_mask] = np.stack(
        [(R_non_zero[:, 2, 1] - R_non_zero[:, 1, 2]) / (2 * sin_theta_non_zero),
         (R_non_zero[:, 0, 2] - R_non_zero[:, 2, 0]) / (2 * sin_theta_non_zero),
         (R_non_zero[:, 1, 0] - R_non_zero[:, 0, 1]) / (2 * sin_theta_non_zero)], axis=1)
    rotvec = v * np.expand_dims(theta, axis=1) * np.logical_not(zero_mask)[...,None]

    return rotvec



## -------------------------
def batch_rotMat2Vec_tensor(rotmat:torch.Tensor, strict_mode:bool=True) -> torch.Tensor:
    '''
    Calculates the rotation vectors for a batch of rotation matrices.

    Parameter
    ----
    rotmat : Tensor
        `[N, 3, 3]`, The rotation matrices for the given axis-angle parameters
    strict_mode : bool
        if `True`, raise error when inputs matrix don't pass the orthognal check or det check, else will produce an approximate matrix using SVD.

    Returns
    ----
    rot_vectors : Tensor
        `[N, 3]`, array of N axis-angle vectors

    '''
    _eps = 1e-5

    R_shape = rotmat.shape
    batch_size = R_shape[0]

    if R_shape[1:] != (3, 3):
        raise ValueError("Input matrix should be of shape (n, 3, 3).")

    ortho_check = torch.allclose(rotmat @ rotmat.transpose(-1, -2),
                                 torch.eye(3, dtype=rotmat.dtype, device=rotmat.device),
                                 rtol=_eps, atol=_eps)
    det_check = torch.allclose(torch.det(rotmat),
                               torch.tensor(1., dtype=rotmat.dtype, device=rotmat.device),
                               rtol=_eps, atol=_eps)

    if (not det_check) or (not ortho_check):
        if strict_mode:
            raise ValueError(f"Input rotation matrices' Determinantis is not 1. or not Orthogonal. Check results is det_check {det_check} and ortho_check {ortho_check}.")
        else:
            u, s, vh = torch.linalg.svd(rotmat) # produce an approximate standard rotate matrix using SVD.
            rotmat = u @ vh

    trace = torch.einsum('bii->b', rotmat).clamp(-1., 3.) # make sure that cos_theta values in [-1, 1].

    theta = torch.acos((trace - 1.) / 2.)
    zero_mask = torch.abs(theta) < _eps
    v = torch.zeros((batch_size, 3), dtype=rotmat.dtype, device=rotmat.device)

    R_non_zero = rotmat[~zero_mask]
    theta_non_zero = theta[~zero_mask]
    sin_theta_non_zero = torch.sin(theta_non_zero) + _eps

    v[~zero_mask] = torch.stack(
        [(R_non_zero[:, 2, 1] - R_non_zero[:, 1, 2]) / (2 * sin_theta_non_zero),
         (R_non_zero[:, 0, 2] - R_non_zero[:, 2, 0]) / (2 * sin_theta_non_zero),
         (R_non_zero[:, 1, 0] - R_non_zero[:, 0, 1]) / (2 * sin_theta_non_zero)], dim=1)
    rotvec = v * theta.unsqueeze(1) * (~zero_mask).float()[...,None]

    return rotvec




## -------------------------
def batch_rotR6d2Mat_ndarray(rotr6d:np.ndarray) -> np.ndarray:
    """
    Parameter
    ----
    rotr6d : ndarray
        (B, 6) or (B, 3, 2), Batch of 6-D rotation representations.

    Return:
    ----
    rotmat : ndarray
        Batch of corresponding rotation matrices with shape (B,3,3).

    """

    batch_size = rotr6d.shape[0]
    eps = 1e-8

    rotr6d = rotr6d.reshape(batch_size, 3, 2)

    v1 = rotr6d[..., 0] # vertical column vector
    v2 = rotr6d[..., 1]

    # SO3
    mats_v1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + eps) # equal to F.normalize()
    mats_v2 = (v2 - np.einsum('bi,bi->b', mats_v1, v2)[...,None] * mats_v1)
    mats_v2 = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + eps)
    mats_v3 = np.cross(mats_v1, mats_v2, axis=-1)

    return np.stack([mats_v1, mats_v2, mats_v3], axis=-1)





## --------------------------
def batch_rotR6d2Mat_tensor(rotr6d:torch.Tensor) -> torch.Tensor:
    """
    Parameter
    ----
    rotr6d : Tensor
        (B, 6) or (B, 3, 2), Batch of 6-D rotation representations.

    Return:
    ----
    rotmat : Tensor
        Batch of corresponding rotation matrices with shape (B,3,3).

    >>> mat = torch.tensor([[[ 0.8434, -0.5166,  0.1480],
                             [ 0.3517,  0.7389,  0.5747],
                             [-0.4062, -0.4326,  0.8049]],
                            [[ 0.8951, -0.4274,  0.1268],
                             [ 0.4076,  0.8998,  0.1558],
                             [-0.1807, -0.0878,  0.9796]],
                            [[ 0.9184, -0.3903, -0.0648],
                             [ 0.3690,  0.7861,  0.4958],
                             [-0.1426, -0.4793,  0.8660]]])
        rebuild_mat = batch_rotR6d2Mat_tensor(mat[ :, :, :2])
        rebuild_mat == mat

    """

    batch_size = rotr6d.shape[0]
    eps = 1e-8

    rotr6d = rotr6d.contiguous()
    rotr6d = rotr6d.reshape(batch_size, 3, 2)

    v1 = rotr6d[..., 0]
    v2 = rotr6d[..., 1]

    # SO3
    mats_v1 = F.normalize(v1, dim=-1, eps=eps)
    mats_v2 = (v2 - torch.einsum('bi,bi->b', mats_v1, v2)[...,None] * mats_v1)
    mats_v2 = F.normalize(v2, dim=-1, eps=eps)
    mats_v3 = torch.cross(mats_v1, mats_v2, dim=-1)

    return torch.stack([mats_v1, mats_v2, mats_v3], dim=-1)



# --------------------------
def batch_rotMat2R6d_ndarray(rotmat: np.ndarray):
    """
    Parameter
    ----
    rotmat : ndarray
        `[B, 3, 3]`, Batch of rotation matrices 6-D rotation representations.

    Return:
    ----
    rot6d : ndarray
        `[B, 3, 2]`, Batch of rotation 6D
    """
    return rotmat[..., :2]



# --------------------------
def batch_rotMat2R6d_tensor(rotmat: torch.Tensor):
    """
    Parameter
    ----
    rotmat : tensor
        `[B, 3, 3]`, Batch of rotation matrices 6-D rotation representations.

    Return:
    ----
    rot6d : tensor
        `[B, 3, 2]`, Batch of rotation 6D
    """
    return rotmat[..., :2]



# modified from 2023_NIPS_GenPose, checked.
# fixme: should be checked again for batch computation.
def average_quaternion(Q, W=None):
    """calculate the average quaternion of the multiple quaternions along the -2 dimension
    Args:
        Q: (B, ..., N, 4)
        weights: (B, ..., N). Defaults to None.
    Returns:
        oriented_q_avg: average quaternion, (B, ..., 4)
    """
    shape = Q.shape
    assert shape[-1] == 4

    if W is None:
        W = torch.ones_like(Q[..., 0])
    else:
        assert shape[:-1] == W.shape

    weight_sum = W.sum(dim=-1, keepdim=True)
    oriented_Q = ((Q[..., 0:1] > 0).float() - 0.5) * 2 * Q
    A = torch.einsum("...ni,...nj->...nij", oriented_Q, oriented_Q)
    A = torch.sum(torch.einsum("...nij,...n->...nij", (A, W)), -3)
    A /= weight_sum.reshape(*shape[:-2], 1, 1)

    q_avg = torch.linalg.eigh(A)[1][..., -1]
    oriented_q_avg = ((q_avg[..., 0:1] > 0).float() - 0.5) * 2 * q_avg
    return oriented_q_avg


if __name__ == '__main__':
    # to verify the functions

    import cv2

    rotvec = np.array([[ 0.224,  1.832, -0.768],
                       [-0.396, -0.356,  1.294]]).reshape(2, 3)
    rotmat = np.zeros([2, 3, 3])
    for i in range(2):
        rotmat[i], _ = cv2.Rodrigues(rotvec[i].reshape(3, 1))

    print('rotvec:\n', rotvec)
    print('rotmat:\n', rotmat)
    print()


    print('rmt-rvc ndarray version test:')

    rmt_hat = batch_rotVec2Mat_ndarray(rotvec)
    rvc_hat = batch_rotMat2Vec_ndarray(rmt_hat)

    print('rvc err: ', np.sum(np.abs(rvc_hat - rotvec)))
    print('rmc err: ', np.sum(np.abs(rmt_hat - rotmat)))
    print()


    print('rmt-rvc tensor version test:')
    rmt_hat = batch_rotVec2Mat_tensor(torch.from_numpy(rotvec))
    rvc_hat = batch_rotMat2Vec_tensor(rmt_hat)

    print('rvc err: ', np.sum(np.abs(rvc_hat.numpy() - rotvec)))
    print('rmc err: ', np.sum(np.abs(rmt_hat.numpy() - rotmat)))
    print()


    print('r6d-rmt ndarray test:')
    r6d_hat = batch_rotMat2R6d_ndarray(rotmat)
    rmt_hat = batch_rotR6d2Mat_ndarray(r6d_hat)

    print('rebuild rmt err: ', np.sum(np.abs(rmt_hat - rotmat)))
    print()


    print('r6d-rmt tensor test:')
    r6d_hat = batch_rotMat2R6d_tensor(torch.from_numpy(rotmat))
    rmt_hat = batch_rotR6d2Mat_tensor(r6d_hat)

    print('rebuild rmt err: ', np.sum(np.abs(rmt_hat.numpy() - rotmat)))
    print()


