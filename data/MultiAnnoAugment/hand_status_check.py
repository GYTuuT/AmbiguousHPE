

import torch
from torch import Tensor
import numpy as np

from typing import Tuple, Union, List, Dict, Any
from tools.p3dRender import Pytorch3DRenderer, batch_project_K
from tools.networkUtils import EasyDict as edict


# --------------------------
#           16  12
#       20  |   |   8
#       |   |   |   |
#       19  15  11  7    4
#       |   |   |   |    |
#       18  14  10  6    3
#       |   |   |   |    |
#       17  13  9   5    2
#        \  |   |  /    /
#         \   |   /   1
#           \   /
#             0
# -------------------------


# =================
class ObjectHandScene: # scene with hands, objects and cameras.
    exist_left_hand: bool # whether left hand exists
    exist_right_hand: bool # whether right hand exists

    hand_verts: Tensor # [B, 778, 3] or [B, 778*2, 3], hand vertices, single hand or two hands.
    hand_faces: Tensor # [B, 1538, 3] or [B, 1538*2, 3], hand faces
    hand_j3d: Tensor # [B, 21, 3] or [B, 21*2, 3], hand joints

    hand_gaussian_pts: Tensor # [B, num_gaussian, 21, 3] or [B, num_gaussian, 21*2, 3], 3D hand gaussian points
    hand_gaussian_std: Tensor # [B, 21] or [B, 21*2], standard deviation of each joint gaussian

    obj_verts: Tensor # [B, N, 3], object vertices
    obj_faces: Tensor # [B, F, 3], object faces

    image_size: Tuple[int, int] # (Hi, Wi), size of original image
    bbox_size:  Tuple[int, int] # (Hr, Wr), size of cropped bbox

    cam_K:  Tensor # [B, 4, 4], camera intrinsic matrix, which project the 3D points to 2D image plane
    proj_K: Tensor # [B, 4, 4], projection matrix, which project the 3D points to 2D bbox plane

    # ----------------
    def __init__(self,
                 image_size:Tuple[int, int],
                 bbox_size: Tuple[int, int],

                 cam_K: Tensor,
                 proj_K:Tensor,

                 obj_verts: Tensor=None,
                 obj_faces: Tensor=None,

                 left_hand_verts: Tensor = None, # [B, 778, 3]
                 left_hand_faces: Tensor = None, # [B, 1538, 3]
                 left_hand_j3d:Tensor = None, # [B, 21, 3]

                 right_hand_verts: Tensor = None,
                 right_hand_faces: Tensor = None,
                 right_hand_j3d: Tensor = None):

        self.image_size = image_size
        self.bbox_size = bbox_size
        self.cam_K = cam_K.reshape(-1, 4, 4)
        self.proj_K = proj_K.reshape(-1, 4, 4)

        if (cam_K - proj_K).abs().sum() < 1e-3:
            assert image_size == bbox_size, "image size should be the same as bbox size when cam_K equals to proj_K."

        self.obj_verts = obj_verts
        self.obj_faces = obj_faces

        hand_verts, hand_faces, hand_j3d = [], [], [] # order in [left hand, right hand]
        hand_gaussian_pts, hand_gaussian_std = [], []
        self.exist_left_hand, self.exist_right_hand = False, False

        if left_hand_verts is not None:
            assert (left_hand_faces is not None) and (left_hand_j3d is not None), "left hand faces and joints should be provided."
            self.exist_left_hand = True
            hand_verts.append(left_hand_verts)
            hand_faces.append(left_hand_faces)
            hand_j3d.append(left_hand_j3d)
            l_gaussian_pts, l_gaussian_std = self.construct_hand_gaussian(left_hand_j3d)
            hand_gaussian_pts.append(l_gaussian_pts)
            hand_gaussian_std.append(l_gaussian_std)

        if right_hand_verts is not None:
            assert (right_hand_faces is not None) and (right_hand_j3d is not None), "right hand faces and joints should be provided."
            self.exist_right_hand = True
            hand_verts.append(right_hand_verts)
            hand_faces.append(right_hand_faces if len(hand_faces) == 0 else right_hand_faces + left_hand_verts.shape[1])
            hand_j3d.append(right_hand_j3d)
            r_gaussian_pts, r_gaussian_std = self.construct_hand_gaussian(right_hand_j3d)
            hand_gaussian_pts.append(r_gaussian_pts)
            hand_gaussian_std.append(r_gaussian_std)

        assert len(hand_verts) > 0, "At least one hand should be provided."

        self.hand_verts = torch.cat(hand_verts, dim=-2)
        self.hand_faces = torch.cat(hand_faces, dim=-2)
        self.hand_j3d   = torch.cat(hand_j3d, dim=-2)
        self.hand_gaussian_pts = torch.cat(hand_gaussian_pts, dim=-2) # [B, N, J, 3]
        self.hand_gaussian_std = torch.cat(hand_gaussian_std, dim=-1) # [B, J]


    # ----------------
    def construct_hand_gaussian(self, joint3d:Tensor, num_gaussian:int=1000) -> Tuple:
        """
        Construct 3D hand gaussian from hand joints

        Args:
            joint3d: [B, 21, 3], hand joints
            num_gaussian: int, number of gaussian points
        Returns:
            hand_gaussian_pts: [B, num_gaussian, 21, 3], 3D hand gaussian points
            hand_gaussian_std: [B, 21], standard deviation of each joint
        """

        j3d = joint3d.clone()

        knu_0 = j3d[:, [ 1, 5,  9, 13, 17]] - j3d[:, :1] # [B, 5, 3]
        knu_1 = j3d[:, [ 2, 6, 10, 14, 18]] - j3d[:, [ 1, 5,  9, 13, 17]]
        knu_2 = j3d[:, [ 3, 7, 11, 15, 19]] - j3d[:, [ 2, 6, 10, 14, 18]]
        knu_3 = j3d[:, [ 4, 8, 12, 16, 20]] - j3d[:, [ 3, 7, 11, 15, 19]]
        j3d[:, [ 4, 8, 12, 16, 20]] = j3d[:, [ 4, 8, 12, 16, 20]] - knu_3 * 0.25 # 替换新的指尖

        knu_0_len = knu_0.square().sum(dim=-1).sqrt() # Knuckle length, [B, 5]
        knu_1_len = knu_1.square().sum(dim=-1).sqrt()
        knu_2_len = knu_2.square().sum(dim=-1).sqrt()
        knu_3_len = knu_3.square().sum(dim=-1).sqrt()
        knu_lens = torch.stack([knu_0_len, knu_1_len, knu_2_len, knu_3_len], dim=-1) # [B, 5, 4]

        knu_std_scale = torch.tensor([[0.60, 0.33, 0.40, 0.25], # thumb, gaussian std / knuckle length
                                      [0.12, 0.32, 0.40, 0.30], # index
                                      [0.12, 0.32, 0.40, 0.28], # middle
                                      [0.12, 0.35, 0.37, 0.28], # ring
                                      [0.12, 0.40, 0.42, 0.28]], # pinky
                                      device=joint3d.device).reshape(1, 5, 4)

        kun_std = (knu_lens * knu_std_scale / 3).reshape(-1, 20, 1) # [B, 20, 1], no root joint, / 3 means 3_theta range of gaussian distribution, which has 99.7% probability
        root_std = (knu_lens[..., 0].mean(-1) * 0.33 / 3).reshape(-1, 1, 1) # [B, 1, 1], root joint, 0.33 is a super parameter.

        hand_gaussian_std = torch.cat([root_std, kun_std], dim=1) # [B, 21, 1]
        hand_gaussian_pts = torch.normal(mean=j3d.unsqueeze(1).repeat_interleave(num_gaussian, dim=1), # [B, num_gaussian, 21, 3]
                                         std=hand_gaussian_std.unsqueeze(1).repeat_interleave(num_gaussian, dim=1))

        return hand_gaussian_pts.reshape(-1, num_gaussian, 21, 3), hand_gaussian_std.reshape(-1, 21)




# =================
class HandStatusCheck: # hand status determined

    renderer: Pytorch3DRenderer # renderer for rendering the scene
    scene: ObjectHandScene # scene with hands, objects and cameras.

    # ----------------
    def __init__(self, renderer:Pytorch3DRenderer, scene:ObjectHandScene):

        assert renderer.image_size == scene.bbox_size, "image size should be the same."

        self.renderer = renderer
        self.scene = scene

    # ----------------
    @torch.no_grad()
    def rendering_scene(self) -> Tensor:
        """
        Check the visibility of hand joints, including self-occlusion and occlusion by objects.

        Returns:
            normal_map: [B, H, W, 3], normal map of rendered scene.
        """
        min_obj_vert_idx = self.scene.hand_verts.shape[1] # the start index of rendered vertices that belong to objects

        scene_verts, scene_faces = [self.scene.hand_verts], [self.scene.hand_faces]
        if self.scene.obj_verts is not None:
            scene_verts.append(self.scene.obj_verts)
            scene_faces.append(self.scene.obj_faces + min_obj_vert_idx)
        scene_verts, scene_faces = torch.cat(scene_verts, dim=-2), torch.cat(scene_faces, dim=-2)
        projector = self.renderer.create_perspective_cameras(K=self.scene.proj_K, R=None, T=None)
        normal_map = self.renderer.get_normalMap(scene_verts, scene_faces, projector)

        return normal_map

    # ----------------
    @torch.no_grad()
    def check_joint_visibility(self, return_mask:bool=False) -> Union[Tensor, Tuple]:
        """
        Check the visibility of hand joints, including self-occlusion and occlusion by objects.

        Args:
            return_mask: bool, whether return the mask of hand part and object part.
            return_nmap: bool, whether return the normal map of rendered scene.
        Returns:
            joint_vis: [B, 21], joint visibility, 0 for invisible, 1 for visible.
            hand_mask: [B, H, W], mask of hand part and object part, 1.0 for hand part
            obj_mask:  [B, H, W], mask of object part, 1.0 for object part
        """
        min_obj_vert_idx = self.scene.hand_verts.shape[1] # the start index of rendered vertices that belong to objects
        min_obj_face_idx = self.scene.hand_faces.shape[1] # the start index of rendered faces that belong to objects

        scene_verts, scene_faces = [self.scene.hand_verts], [self.scene.hand_faces]
        if self.scene.obj_verts is not None:
            scene_verts.append(self.scene.obj_verts)
            scene_faces.append(self.scene.obj_faces + min_obj_vert_idx)
        scene_verts, scene_faces = torch.cat(scene_verts, dim=-2), torch.cat(scene_faces, dim=-2)

        projector = self.renderer.create_perspective_cameras(K=self.scene.proj_K, R=None, T=None)
        rast_out = self.renderer.get_rasterResults(scene_verts, scene_faces, projector)

        scene_depth = rast_out['zbuf'][..., 0] # only consider the topest face, background value is -1.0
        scene_depth[scene_depth <= -0.99] = -1000.0 # set background value to -1000

        pix_2_face = rast_out['pix_to_face'][..., 0] # [B, H, W], segment the rendered depth into different part
        if self.scene.exist_left_hand and self.scene.exist_right_hand:
            l_hand_depth = torch.where((pix_2_face > 0.)   & (pix_2_face < 1538), scene_depth, -1000.)
            r_hand_depth = torch.where((pix_2_face > 1538) & (pix_2_face < min_obj_face_idx), scene_depth, -1000.)
        elif self.scene.exist_left_hand and not self.scene.exist_right_hand:
            l_hand_depth = torch.where((pix_2_face > 0.) & (pix_2_face < min_obj_face_idx), scene_depth, -1000.)
            r_hand_depth = torch.ones_like(l_hand_depth) * -1000.
        elif not self.scene.exist_left_hand and self.scene.exist_right_hand:
            r_hand_depth = torch.where((pix_2_face > 0.) & (pix_2_face < min_obj_face_idx), scene_depth, -1000.)
            l_hand_depth = torch.ones_like(r_hand_depth) * -1000.
        else:
            raise ValueError("At least one hand should be provided.")

        B, N, J, _ = self.scene.hand_gaussian_pts.shape
        hand_gaussian_pts2d = batch_project_K(self.scene.hand_gaussian_pts.reshape(B, N*J, 3), self.scene.proj_K) # [B, N*J, 2]
        batch_ind = torch.arange(B, device=hand_gaussian_pts2d.device).reshape(-1, 1).expand(-1, N*J)
        sampling_grid = hand_gaussian_pts2d.round().long()
        sampling_grid[..., 0] = torch.clamp(sampling_grid[..., 0], 0, self.scene.bbox_size[1] - 1)
        sampling_grid[..., 1] = torch.clamp(sampling_grid[..., 1], 0, self.scene.bbox_size[0] - 1)
        l_sampled_depth = l_hand_depth[batch_ind, sampling_grid[..., 1], sampling_grid[..., 0]].reshape(B, N, J)
        r_sampled_depth = r_hand_depth[batch_ind, sampling_grid[..., 1], sampling_grid[..., 0]].reshape(B, N, J)

        if self.scene.exist_left_hand and self.scene.exist_right_hand:
            sampled_depth = torch.cat([l_sampled_depth[...,:21], r_sampled_depth[...,21:]], dim=-1) # [B, N, 42]
        elif self.scene.exist_left_hand and not self.scene.exist_right_hand:
            sampled_depth = l_sampled_depth # [B, N, 21]
        elif not self.scene.exist_left_hand and self.scene.exist_right_hand:
            sampled_depth = r_sampled_depth
        else:
            raise ValueError("At least one hand should be provided.")

        # ````````occlusion determination`````````
        rate_thresh_for_occluded = 0.80

        # 1. occluded by objects, determined as occlusion if 80% of prejected gaussian points are invisible.
        occluded_by_obj = ((sampled_depth < 0.0).sum(dim=1) / N) > rate_thresh_for_occluded # [B, J], true for occluded.

        # 2. occluded by self, determined as occlusion if sampled depth is less than real depth.
        depth_dist_thresh = self.scene.hand_gaussian_std.reshape(B, 1, J) * 3.0 * 2 # 3_sigma diameter of 3D gaussian distribution
        valid = (sampled_depth > 0.0) * 1.0 # [B, N, J], only consider the gasussian pts in valid hand foreground.
        occluded_by_self = (self.scene.hand_gaussian_pts[..., -1] - sampled_depth) > depth_dist_thresh # [B, N, J]
        occluded_by_self = ((occluded_by_self * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-5)) > rate_thresh_for_occluded # [B, J]

        joint_occ = torch.logical_or(occluded_by_obj, occluded_by_self) # [B, J], true for occluded.
        joint_vis = joint_occ.logical_not() # [B, J], true for visible.

        if return_mask:
            hand_mask = torch.where((pix_2_face > 0.) & (pix_2_face < min_obj_face_idx), 1.0, 0.0) # maybe one hand or two hands.
            obj_mask  = torch.where((pix_2_face >= min_obj_face_idx), 1.0, 0.0)

        if return_mask:
            return joint_vis, hand_mask, obj_mask
        else:
            return joint_vis

    # ----------------
    @torch.no_grad()
    def check_joint_outside_image(self) -> Tensor:
        """
        Check whether the hand joints are outside the image.

        Returns:
            joint_outside: [B, 21], true for outside, false for inside.
        """
        rate_thresh_for_outside = 0.80

        B, N, J, _ = self.scene.hand_gaussian_pts.shape
        hand_gaussian_pts2d = batch_project_K(self.scene.hand_gaussian_pts.reshape(B, N*J, 3), self.scene.cam_K) # [B, N*J, 2]

        imgH, imgW = self.scene.image_size
        is_inside_img = (hand_gaussian_pts2d[..., 0] >= 0) & (hand_gaussian_pts2d[..., 0] < imgW) & \
                        (hand_gaussian_pts2d[..., 1] >= 0) & (hand_gaussian_pts2d[..., 1] < imgH) # [B, N*J]
        inside_rate = is_inside_img.reshape(-1, N, J).float().sum(dim=1) / N
        joint_outside = inside_rate < (1 - rate_thresh_for_outside) # [B, 21], true for outside, false for inside.

        return joint_outside

    # ----------------
    @torch.no_grad()
    def check_joint_collision(self) -> Tensor:
        """
        Check whether the hand joints are colliding with each other, using Bhattacharyya distance.

        Returns:
            joint_collided: [B, 42], true for collision, false for no collision.
        """
        gaussian_joints = self.scene.hand_gaussian_pts # [B, N, J, 3]
        B, N, J, V = gaussian_joints.shape

        # 1. mu and sigma of gaussian hands
        samples = gaussian_joints.clone().permute(0, 2, 1, 3) # -> [B, K, N, V]
        mu = samples.mean(dim=-2, keepdim=True) # [B, K, 1, V]
        sigma = (samples - mu).unsqueeze(dim=-1) # [B, K, N, V, 1]
        sigma = (sigma @ sigma.transpose(-1, -2)).sum(dim=-3) / (N - 1) # [B, K, V, V]

        # 2. calculate Bhattacharyya distance
        mu_p = mu.clone().detach().reshape(B, J, 1, 1, V)
        mu_q = mu.clone().detach().reshape(B, 1, J, 1, V)
        sigma_p = sigma.clone().detach().reshape(B, J, 1, V, V)
        sigma_q = sigma.clone().detach().reshape(B, 1, J, V, V)

        sigma_pq = 0.5 * (sigma_p + sigma_q) # [B, K, K, V, V]
        part1 = torch.log(sigma_pq.det() / (sigma_p @ sigma_q).det().sqrt()) # [B, K, K]
        part2 = ((mu_p - mu_q) @ sigma_pq.inverse() @ (mu_p - mu_q).transpose(-1, -2)) # [B, K, K, 1, 1]
        bhatta_dist = 0.5 * part1 + 0.125 * part2.reshape(B, J, J)

        # 3. check collision
        triu_ind = torch.triu_indices(row=21, col=21, offset=1, device=gaussian_joints.device)
        joint_collided = (bhatta_dist[:, triu_ind[0], triu_ind[1]] < 2.25).any(dim=-1) # [B,], 2.25 is a super parameter.

        return joint_collided # [B, ]


    ## ------------------
    @staticmethod
    def determine_finger_occlusion_score(J_vis:Tensor): # [B, 21]
        """ 根据关节的遮挡情况计算手指的遮挡分数, 表明手指的歧义性大小

        每个手指的关节层级定义: root --> 0 --> 1 --> 2 --> 3. 在 MANO 中, 层级 0 只由 root 代表的全局位姿决定, 因此不纳入考虑范围. 而根据运动学反解, 后一级的线索可以大致推断前一级的位姿情况, 因此歧义性分数的规则如下:

        1) 遮挡: [1, 2, 3], 未遮挡: [],     歧义分数: 1.00 ;
        2) 遮挡: [2, 3],    未遮挡: [1],    歧义分数: 0.80 ;
        3) 遮挡: [1, 3],    未遮挡: [2],    歧义分数: 0.60 ;
        4) 遮挡: [3],       未遮挡: [1, 2], 歧义分数: 0.40 ;
        5) 遮挡: [1, 2],    未遮挡: [3],    歧义分数: 0.20 ;
        6) 其他情况, 歧义分数: 0.00

        """

        J_invis = J_vis.logical_not()
        J_invis = J_invis[:, 1:].reshape(-1, 5, 4) # 手指关节的遮挡情况

        F_occlu = torch.zeros([len(J_invis), 5], device=J_invis.device) # [B, 5]
        F_occlu[J_invis[..., [1, 2, 3]].all(dim=-1)] = 1.00 # 规则 1
        F_occlu[J_invis[..., [2, 3]].all(dim=-1) & (~ J_invis)[..., [1]].all(dim=-1)] = 0.80 # 规则 2
        F_occlu[J_invis[..., [1, 3]].all(dim=-1) & (~ J_invis)[..., [2]].all(dim=-1)] = 0.60 # 规则 3
        F_occlu[J_invis[..., [3]].all(dim=-1) & (~ J_invis)[..., [1, 2]].all(dim=-1)] = 0.40 # 规则 4
        F_occlu[J_invis[..., [1, 2]].all(dim=-1) & (~ J_invis)[..., [3]].all(dim=-1)] = 0.20 # 规则 5

        return F_occlu # [B, 5]
