
"""
Custom Manolayers

LastChange: 2023.12.29
Author: GeYuting

"""


"""
(1) As the official 'smplx' and popular 'manopth' packages has 'flat_pose_mean' option, we REMOVE it in our custom manolayer, in order to keep the consistent between rotvec (axis-angle) inputs or rotmat inputs.

(2)  In original manolayer ('smplx' and 'manopth'), Global Rotation will be firstly applied to hand mesh and then transform the fingers' rotation. But in our achieve, it can be applied at the end to avoid confusing translation when do some gemetric transform on Mano mesh, keeping `R @ Mano(r) + T == Mano(R @ r, T)`.
NOTE  When apply global rotation at last, the MANO TRANS anno of many existing dataset should be REPAIRED.

(3) Joints order is different from 'smplx', it's in a more nature order.

"""


"""
MANO_PARTITION_COLOR refer to 'Real-time Pose and Shape Reconstruction of Two Interacting Hands With a Single Depth Camera, SIGGRAPH 2019', https://handtracker.mpi-inf.mpg.de/projects/TwoHands/.
"""

__all__ = ["RotMatrixManolayer",
           "RotVectorManolayer",
           "ManoPcalayer",
           "convert_fingerOrder"]


import os.path as osp
import pickle
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor

#

# content of manolayer's attribute 'pose mean'.
# -----------------------------------------
Mano_Rotvec_Mean_Right = \
[ 0.        ,  0.        ,  0.        ,  0.11167871, -0.04289218,  0.41644183,
  0.10881133,  0.06598568,  0.75622000, -0.09639297,  0.09091566,  0.18845929,
 -0.11809504, -0.05094385,  0.52958450, -0.14369841, -0.05524170,  0.70485710,
 -0.01918292,  0.09233685,  0.33791350, -0.45703298,  0.19628395,  0.62545750,
 -0.21465237,  0.06599829,  0.50689423, -0.36972436,  0.06034463,  0.07949023,
 -0.14186970,  0.08585263,  0.63552827, -0.30334160,  0.05788098,  0.63138920,
 -0.17612089,  0.13209307,  0.37335458,  0.85096430, -0.27692273,  0.09154807,
 -0.49983943, -0.02655647, -0.05288088,  0.53555920, -0.04596104,  0.27735803,]


Mano_Rotvec_Mean_Left = \
[ 0.        ,  0.        ,  0.        ,  0.11167871,  0.04289218, -0.41644183,
  0.10881133, -0.06598568, -0.75622000, -0.09639297, -0.09091566, -0.18845929,
 -0.11809504,  0.05094385, -0.52958450, -0.14369841,  0.05524170, -0.70485710,
 -0.01918292, -0.09233685, -0.33791350, -0.45703298, -0.19628395, -0.62545750,
 -0.21465237, -0.06599829, -0.50689423, -0.36972436, -0.06034463, -0.07949023,
 -0.14186970, -0.08585263, -0.63552827, -0.30334160, -0.05788098, -0.63138920,
 -0.17612089, -0.13209307, -0.37335458,  0.85096430,  0.27692273, -0.09154807,
 -0.49983943,  0.02655647,  0.05288088,  0.53555920,  0.04596104, -0.27735803,]
# ----------------------------------------


# Offical MANO Joints Order ('smplx')     |         Custom (Nature) Joints Order
#   ------------------------------        |        ------------------------------
#          19    18                       |                16    12
#      20   |    |    17                  |           20    |    |    8
#      |    |    |    |                   |            |    |    |    |
#      9   12    6    3   16              |           19   15    11   7    4
#      |    |    |    |    |              |            |    |    |    |    |
#      8   11    5    2   15              |           18   14    10   6    3
#      |    |    |    |    |              |            |    |    |    |    |
#      7   10    4    1   14              |           17   13    9    5   2
#       \   |    |   /   /                |             \   |    |   /   /
#         \  \   /  /  13                 |               \  \   /  /   1
#           \   /                         |                 \   /
#             0                           |                   0
#   -------------------------------       |        ------------------------------


## =================================
class ManoBasic(torch.nn.Module):
    """ NOTE
    Using the NATURE Joints Order.
    """
    def __init__(self,
                 model_path: str='.processed_mano',
                 hand_type: str='right',
                 fix_left_shapedirs: bool=True,
                 apply_global_last: bool=True, # default apply global transformation at last
                 **kwargs) -> None:

        super().__init__()

        self.hand_type = hand_type
        assert hand_type in ['right', 'left'], 'Wrong hand type.'

        model_name = 'MANO_LEFT.pkl' if hand_type =='left' else 'MANO_RIGHT.pkl'
        assert osp.exists(osp.join(model_path, model_name)), 'Wrong MANO model path.'

        model_datas = pickle.load(open(osp.join(model_path, model_name), 'rb'))

        # Load intermediate values
        self.register_buffer('faces',
                             model_datas['faces'].to(dtype=torch.long))
        self.register_buffer('J_regressor',
                             model_datas['J_regressor'].to(dtype=torch.float32))
        self.register_buffer('kintree_table',
                             model_datas['kintree_table'].to(dtype=torch.long))
        self.register_buffer('lbs_weights',
                             model_datas['lbs_weights'].to(dtype=torch.float32))
        self.register_buffer('verts_template',
                             model_datas['verts_template'].to(dtype=torch.float32))
        self.register_buffer('posedirs',
                             model_datas['posedirs'].to(dtype=torch.float32))
        self.register_buffer('shapedirs',
                             model_datas['shapedirs'].to(dtype=torch.float32))
        self.register_buffer('pose_mean',
                             model_datas['pose_mean'].to(dtype=torch.float32))

        # left mano bug fixed: https://github.com/vchoutas/smplx/issues/48, some datasets considered this when some others did not.
        if fix_left_shapedirs and hand_type == 'left':
            self.shapedirs[:,0,:] *= -1

        self.apply_global_last = apply_global_last

        # set fingertips using specific verts
        # follow: https://github.com/hassony2/manopth/blob/master/manopth/manolayer.py
        extra_tips = {'thumb':745, 'index':317, 'middle':444, 'ring':556, 'pinky':673,}
        self.register_buffer('extra_tips_idxs',
                             torch.tensor(list(extra_tips.values()), dtype=torch.long))

    # -----------
    def name(self) -> str:
        return 'ManoBasic'


    # -----------
    def forward(self,
                global_rmt:Tensor, finger_rmt:Tensor,
                shape:Tensor, trans:Tensor=None) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----
        global_rmt : Tensor
            `[B, 1, 3, 3]`, global rotation matrix.
        finger_rmt : Tensor
            `[B, 15, 3, 3]`, finger rotation matrices.
        shape : Tensor
            `[B, 10]`, shape params of mano.
        trans
            `[B, 3]`, global translation of mano.

        Returns
        ----
        verts : Tensor
            `[B, 778, 3]`, mano vertices.
        joints : Tensor
            `[B, 21, 3]`, mano joints.

        """

        assert global_rmt.shape[1:] == (1, 3, 3), "Expected global_rmt to have shape (B, 1, 3, 3)"
        assert finger_rmt.shape[1:] == (15, 3, 3), "Expected finger_rmt to have shape (B, 15, 3, 3)"
        assert shape.shape[1:] == (10,), "Expected shape to have shape (B, 10)"
        if trans is not None:
            assert trans.shape[1:] == (3,), "Expected trans to have shape (B, 3)"

        dtype, device = self.verts_template.dtype, self.verts_template.device

        global_rmt = global_rmt.to(dtype=dtype, device=device)
        finger_rmt = finger_rmt.to(dtype=dtype, device=device)
        shape = shape.to(dtype=dtype, device=device)
        trans = torch.zeros([1, 3]) if trans is None else trans
        trans = trans.to(dtype=dtype, device=device)

        verts, joints = self.lbs_mano(pose=torch.cat([global_rmt, finger_rmt], dim=1),
                                      shape=shape)

        verts += trans.reshape(-1, 1, 3)  # [B, 778, 3]
        joints += trans.reshape(-1, 1, 3) # [B, 16, 3]

        joints = torch.cat([joints, verts[ :, self.extra_tips_idxs, :]], dim=1) # add finger tips
        joints = joints[:, [0, 13, 14, 15, 16,
                                1,  2,  3, 17,
                                4,  5,  6, 18,
                               10, 11, 12, 19,
                                7,  8,  9, 20], :] # to nature order

        return verts, joints


    # -----------
    def lbs_mano(self, pose:Tensor, shape:Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        pose : torch.tensor
            `[B, 16, 3, 3]`, The pose parameters in rotmat.
        shape : torch.tensor
            `[B, 10]`, The tensor of shape parameters

        Returns
        -------
        verts: torch.tensor
            `[B, 778, 3]`,  The vertices of the mesh after applying the shape and pose displacements.
        joints: torch.tensor
            `[B, 16, 3]`, The joints of the model
        '''

        B = max(shape.shape[0], pose.shape[0])
        Nv, Nj = 778, 16
        device, dtype = shape.device, shape.dtype

        I = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3)

        # 1. Apply shape skinning and get joints
        V = self.verts_template[None, ...] + torch.einsum('bl,vkl->bvk', [shape, self.shapedirs]) # [B, Nv, 3]
        J = torch.einsum('bvk,jv->bjk', [V, self.J_regressor]) # [B, Nj, 3]

        # 2. Split the pose rotation
        global_rmt = pose[:, 0, ...] # [B, 3, 3]
        finger_rmt = pose[:, 1:, ...] # [B, 15, 3, 3]

        # 3. Apply posed skinning, [B, 15*3*3] @  [15*3*3, Nv*3] = [B, Nv*3]
        pose_feature = (finger_rmt - I.clone()).reshape(B, -1) # [B, 135]
        V = V + (pose_feature @ self.posedirs).reshape(B, -1, 3) # [B, V, 3]

        # 4. Compute the transformed joint and transform chain
        if self.apply_global_last:
            transform_rmt = torch.cat([I.clone().repeat(B, 1, 1, 1), finger_rmt], dim=1) # [B, Nj, 3, 3]
        elif not self.apply_global_last:
            transform_rmt = pose
        else:
            ...
        J_rel = J.clone()
        J_rel[:, 1:] = J_rel[:, 1:] - J[:, self.kintree_table[1:]] # [B, Nj, 3], relative in joint level.

        M = self.transform_mat(R=transform_rmt.reshape(-1, 3, 3),
                               T=J_rel.reshape(-1, 3, 1)).reshape(-1, Nj, 4, 4)
        root_M, finger_M = M[:, :1], M[:, 1:].reshape(-1, 5, 3, 4, 4) # [B, F, J_per_F, 4, 4]

        finger_M_0 = root_M.repeat(1, 5, 1, 1) @ finger_M[ :, :, 0] # joint level 1
        finger_M_1 = finger_M_0 @ finger_M[ :, :, 1] # joint level 2
        finger_M_2 = finger_M_1 @ finger_M[ :, :, 2] # joint level 3

        finger_M = torch.stack([finger_M_0, finger_M_1, finger_M_2], dim=2)
        M = torch.cat([root_M, finger_M.reshape(-1, 15, 4, 4)], dim=1) # transform chain, [B, Nj, 4, 4]

        J_transformed = M[ :, :, :3, 3].clone() # [B, Nj, 3],
        M[..., :3, 3:] -= M[..., :3, :3].clone() @ J[..., None] # update the relative trans for V

        # 5. Apply lbs skinning
        W = self.lbs_weights[None, ...].repeat(B, 1, 1) # [B, V, J]
        M = (W @ M.reshape(B, Nj, 16)).reshape(B, Nv, 4, 4) # [B, V, J] @ [B, J, 4*4] -> [B, V, 4*4]
        V_transformed = (M[..., :3, :3] @ V[..., None] + M[..., :3, 3:])[..., 0] # R @ V + t, [B, V, 3]

        # 6. Apply global ratation
        if self.apply_global_last:
            J_transformed = J_transformed @ global_rmt.transpose(-1, -2)
            V_transformed = V_transformed @ global_rmt.transpose(-1, -2)

        return V_transformed, J_transformed


    ## ----------------------
    @staticmethod
    def batch_rotVec2Mat(rotvec:Tensor) -> Tensor: # [B, 3] -> [B, 3, 3]
        """ Calculates the rotation matrices from a batch of rotation vectors
        """
        batch_size = rotvec.shape[0]
        device, dtype = rotvec.device, rotvec.dtype

        angle = torch.norm(rotvec + 1e-8, dim=1, keepdim=True)
        rot_dir = rotvec / (angle + 1e-8)

        cos = torch.unsqueeze(torch.cos(angle), dim=1)
        sin = torch.unsqueeze(torch.sin(angle), dim=1)

        # Bx1 arrays
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        K = torch.cat([zeros, -rz,  ry,
                        rz, zeros, -rx,
                       -ry,  rx, zeros], dim=1).view((batch_size, 3, 3))

        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
        rotmat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

        return rotmat


    ## ----------------------
    @staticmethod
    def transform_mat(R:Tensor,
                      T:Tensor) -> Tensor: # [B,3,3] + [B,3,1] -> [B,4,4]
        """ Creates a batch of transformation matrices
        """
        # No padding left or right, only add an extra row
        return torch.cat([F.pad(R, [0, 0, 0, 1]),
                          F.pad(T, [0, 0, 0, 1], value=1)], dim=2)



## ==================
class RotMatrixManolayer(ManoBasic):
    def __init__(self, model_path:str, hand_type:str, **kwargs) -> None:
        super(RotMatrixManolayer, self).__init__(model_path, hand_type, **kwargs)

    # -------
    def name(self) -> str:
        return 'RotMatrixManolayer'

    # -------
    def forward(self,
                global_rmt:Tensor, finger_rmt:Tensor,
                shape:Tensor, trans:Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        See ManoBasic's forward discription.
        """

        return super().forward(global_rmt, finger_rmt, shape, trans)



## ==================
class RotVectorManolayer(ManoBasic):
    def __init__(self, model_path:str, hand_type:str, **kwargs) -> None:
        super().__init__(model_path, hand_type, **kwargs)

    # --------
    def name(self) -> str:
        return 'RotVectorManoLayer'

    # ---------
    def forward(self,
                global_rvc: Tensor, finger_rvc: Tensor,
                shape: Tensor, trans: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        global_rvc : Tensor
            `[B, 1, 3]`, global rotation vector (axis-angle)
        finger_rvc : Tensor
            `[B, 15, 3]`, finger rotation vectors (axis-angle)

        Others see ManoBasic's forward discription.
        """

        global_rmt = self.batch_rotVec2Mat(global_rvc.reshape(-1, 3)).reshape(-1, 1, 3, 3)
        finger_rmt = self.batch_rotVec2Mat(finger_rvc.reshape(-1, 3)).reshape(-1, 15, 3, 3)

        return super().forward(global_rmt, finger_rmt, shape, trans)




## ===============================
class ManoPcalayer(torch.nn.Module):
    def __init__(self,
        model_path: str='.mano_models/processed_mano',
        **kwargs) -> None:
        super().__init__()

        r_model = pickle.load(open(osp.join(model_path, 'MANO_RIGHT.pkl'), 'rb'))
        l_model = pickle.load(open(osp.join(model_path, 'MANO_LEFT.pkl'), 'rb'))

        r_pca_mat = r_model['hands_pca_components'].clone().detach().to(dtype=torch.float32)
        l_pca_mat = l_model['hands_pca_components'].clone().detach().to(dtype=torch.float32)

        self.register_buffer('r_pca_mat', r_pca_mat.clone()) # pca -> pose
        self.register_buffer('l_pca_mat', l_pca_mat.clone())
        self.register_buffer('r_pca_mat_proj', r_pca_mat.clone().inverse()) # pose -> pca
        self.register_buffer('l_pca_mat_proj', l_pca_mat.clone().inverse())


    def rotvec2pca(self, rotvec:Tensor, hand_type:str, n_comps:int):
        """
        Transform rotation vector (axis-angle) to pca code.

        Parameter
        ----
        rotvec : Tensor
            `[N, 45]`, joint rotation axis-angle of the hand except root joint, NOTE without 'pose mean'.
        hand_type : str
            `'right' or 'left'`.
        n_comps : int
            Num of target pca code components.

        Returns
        ----
        pca_code : Tensor
            `[N, n_comps]`.

        """
        assert n_comps <= 45, 'Wrong Num of PCA Components.'

        if hand_type == 'right':
            pca_code = torch.einsum('bi,ij->bj', rotvec, self.r_pca_mat_proj[...,:n_comps])
        elif hand_type == 'left':
            pca_code = torch.einsum('bi,ij->bj', rotvec, self.l_pca_mat_proj[...,:n_comps])
        else:
            raise ValueError('Invalid Hand Type.')

        return pca_code


    def pca2rotvec(self, pca_code:Tensor, hand_type:str):
        """
        Transform pca code to rotation vector (axis-angle).

        Parameter
        ----
        pca_code : Tensor
            `[N, n_comps]`.
        hand_type : str
            `'right' or 'left'`.

        Returns
        ----
        rotvec : Tensor
            `[N, 45]`, joint rotation axis-angle of the hand except root joint, NOTE without 'pose mean'.

        """
        n_comps = pca_code.shape[1]
        assert n_comps <=45, 'Wrong input shape of PCA code.'

        if hand_type == 'right':
            rotvec = torch.einsum('bi,ij->bj', pca_code, self.r_pca_mat[:n_comps])
        elif hand_type == 'left':
            rotvec = torch.einsum('bi,ij->bj', pca_code, self.l_pca_mat[:n_comps])
        else:
            raise ValueError('Invalid Hand Type.')

        return rotvec



## ----------------
def convert_fingerOrder(inputs:Union[ndarray, Tensor], flag:str=''):
    """convert the finger order between mano and nature.

    Params
    ----
    inputs : ndarray or tensor
        `[B, 5, ...]`, [batch_dim, finger_dim, ...].
    flag : str
        the flag of converting direction, `mano2nature` or `nature2mano`.

    """
    assert flag in ['mano2nature', 'nature2mano'], 'Invalid flag.'

    if flag == 'mano2nature':
        return inputs[ :, [4, 0, 1, 3, 2], ...]
    elif flag == 'nature2mano':
        return inputs[ :, [1, 2, 4, 3, 0], ...]
    else:
        ...





if __name__ == '__main__':

    a = RotVectorManolayer(model_path='mano/mano_models/processed_mano',
                           hand_type='right')

    print(a.kintree_table)
    exit()

    global_rot = torch.randn([1, 1, 3, 3])
    finger_rot = torch.randn([1, 15, 3, 3])
    shape = torch.randn([1, 10])

    a(global_rot, finger_rot, shape)
