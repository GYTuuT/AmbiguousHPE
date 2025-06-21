import os
import sys

sys.path.append(os.getcwd())

import os.path as osp
import pickle
import random
from copy import deepcopy
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import Dataset

from tools.dataTransform import (AugmentPipe, apply_crop_resize,
                                 produce_points2d_bbox, rmt_2_rvc, rvc_2_rmt,
                                 transform2d)


# ------------------------------
def _padding(x:np.ndarray, pad_value:float=1.0):
    N,_ = x.shape
    x_pad = np.ones((N, 1), dtype=np.float32) * pad_value
    return np.concatenate([x, x_pad], axis=-1)

# ------------------------------
class ArcticFrameDataset(Dataset):
    """ NOTE In Arctic, the Left Hand Use Manolayer with Unfixed Shapedirs. Each image has two hands in it.
    NOTE The augmented multi-hypotheses have some problems.
    """
    def __init__(self,
                 data_dir:str='/root/Workspace/DATASETS/ARCTIC',
                 mode:str='val',
                 resolution:int=256,
                 hand_type:str='right', # 'right', 'left'
                 protocal:str='p1',
                 num_hypotheses:int=100,
                 proportion_hypotheses:int=1.0, # proportion of ambiguous hypotheses
                 decimation:int=4, # decimate the frames for faster training.
                 **kwargs) -> None:

        self.data_dir = data_dir
        self.image_dir = osp.join(data_dir, 'cropped_images')
        self.anno_dir  = osp.join(data_dir, 'custom_annotations')

        assert mode in ['train', 'val'], 'Mode should be train, val or test.'
        self.mode = mode
        self.resolution = resolution
        self.num_hypotheses = num_hypotheses
        self.proportion_hypotheses = proportion_hypotheses

        assert hand_type in ['right', 'left'], 'Hand type should be right, left'
        self.hand_type = hand_type
        self.protocal = protocal

        # index_chain is the frame names ordered in [frm_0_view_0, frm_0_view_1, ..., frm_0_view_7, frm_1_view_0, ...]
        self.index_chains = pickle.load(open(osp.join(self.anno_dir, # frames x views (8)
                                                      f'{mode}_{protocal}_index_chain.pkl'), 'rb'))
        self.index_chains = self.decimate_frames(decimation=decimation)

        self.objects = pickle.load(open(osp.join(self.anno_dir,
                                                 'official_simplified_objects.pkl'), 'rb'))

        self.augment_pipe = AugmentPipe(color_prob=0.5, noise_prob=0.25, geometry_prob=0.5)

    # ------------------
    def __len__(self) -> int:
        return len(self.index_chains)

    # ------------------
    def __getitem__(self, idx:int) -> Dict:
        dtype = torch.float32
        htype = '_r' if self.hand_type == 'right' else '_l'

        frame_data = self.fetch_one_data(idx, get_img=True, get_obj=False)
        frame_data = self.apply_transform(frame_data, do_augment=(self.mode=='train'), do_flip_left=(self.hand_type=='left'))

        mano_rmt = rvc_2_rmt(torch.from_numpy(frame_data[f'mano_rvc{htype}'])).to(dtype=dtype) # [16, 3, 3]

        multi_rvc = np.expand_dims(frame_data[f'mano_rvc{htype}'], axis=0).repeat(self.num_hypotheses, axis=0) # [N, 16, 3]
        if self.proportion_hypotheses > 0.0:
            num_replace = int(self.num_hypotheses * self.proportion_hypotheses) # number of hypotheses using ambigious anno.
            multi_rvc[:num_replace, 1:] = frame_data[f'hand_multi_rvc{htype}'][:num_replace] # random select N_hypo
            np.random.shuffle(multi_rvc) # shuffle the hypotheses
        multi_rvc = multi_rvc + np.random.randn(*multi_rvc.shape) * 0.01 # add noise to hypotheses
        multi_rmt = rvc_2_rmt(torch.from_numpy(multi_rvc).to(dtype=dtype)).reshape(self.num_hypotheses, 16, 3, 3)

        hand_trans = torch.from_numpy(frame_data[f'mano_trans{htype}']).to(dtype=dtype) # [3,]

        outputs = dict(
            is_right = torch.tensor(1 if self.hand_type == 'right' else 0,).to(torch.bool),
            image = torch.from_numpy(frame_data['image'] / 255.).permute(2, 0, 1).to(dtype=dtype), # [3, H, W]

            root_rmt = mano_rmt[:1], # [1, 3, 3]
            hand_rmt = mano_rmt[1:], # [15, 3, 3]
            multi_rmt = multi_rmt[:, 1:], # [N, 15, 3, 3]

            hand_shape = torch.from_numpy(frame_data[f'mano_shape{htype}']).to(dtype=dtype), # [10,]
            hand_trans = hand_trans, # [3,]

            joint3d = torch.from_numpy(frame_data[f'j3d{htype}']).to(dtype=dtype), # [21, 3]
            joint2d = torch.from_numpy(frame_data[f'j2d{htype}']).to(dtype=dtype), # [21, 2]
            joint_vis = torch.from_numpy(frame_data[f'jvis{htype}']).to(torch.bool), # [5, 4]
            joint_valid = torch.from_numpy(frame_data[f'joint_valid{htype}']).to(torch.bool), # [21,]
            hand_valid = torch.tensor(frame_data[f'hand_valid{htype}'].item(),).to(torch.bool), # [1,]

            proj_K = torch.from_numpy(frame_data['proj_K']).to(dtype=dtype), # [3, 3]
        )

        return outputs

    # ----------------
    def apply_transform(self, frame_data:Dict, do_augment:bool=False, do_flip_left:bool=False) -> Dict:
        """Apply the transformation to the frame data.
        """
        # NOTE After augmentation and flipping, the object information becomes invalid.
        # NOTE Geometry augmentation invalidates the global transformation and projection matrix.

        htype = '_r' if self.hand_type == 'right' else '_l'
        # 1) crop and resize the image.
        crop_img, crop_matrix = apply_crop_resize(frame_data['image'], frame_data[f'bbox{htype}'], (self.resolution,)*2)
        frame_data['image'] = crop_img
        frame_data[f'j2d{htype}'] = (crop_matrix[:2, :2] @ frame_data[f'j2d{htype}'].T + crop_matrix[:2, 2:]).T
        frame_data[f'proj_K'] = crop_matrix @ frame_data[f'cameraK'].copy()

        # 2) augmentaion
        if do_augment:
            assert hasattr(self, 'augment_pipe'), 'Augment pipe is not initialized.'
            augment_params = self.augment_pipe.get_augment_params_randomly()
            global_rmt = rvc_2_rmt(frame_data[f'mano_rvc{htype}'][0].copy()) # [3, 3]
            aug_img, (aug_j2d, aug_j3d, aug_global_rotmat), aug_matrix = self.augment_pipe.forward_augment(
                augment_params, frame_data['image'], frame_data[f'j2d{htype}'], frame_data[f'j3d{htype}'], global_rmt)
            frame_data['image'] = aug_img
            frame_data[f'j2d{htype}'] = aug_j2d
            frame_data[f'j3d{htype}'] = aug_j3d
            frame_data[f'mano_rvc{htype}'][0] = rmt_2_rvc(aug_global_rotmat)

        # 3) flip the left hand
        if do_flip_left and self.hand_type == 'left':
            frame_data['image'] = cv2.flip(frame_data['image'], 1)
            frame_data[f'j2d{htype}'][:, 0] = self.resolution - frame_data[f'j2d{htype}'][:, 0]
            frame_data[f'j3d{htype}'][:, 0] = -frame_data[f'j3d{htype}'][:, 0]
            frame_data[f'mano_rvc{htype}'][..., 1:] = frame_data[f'mano_rvc{htype}'][..., 1:] * -1
            frame_data[f'hand_multi_rvc{htype}'][..., 1:] = frame_data[f'hand_multi_rvc{htype}'][..., 1:] * -1

        return frame_data

    # ------------------
    def fetch_one_data(self, data_idx:int, get_img:bool=True, get_obj:bool=True) -> Dict:
        """Fetch the data of a frame.

        Args:
            data_idx (int): the index of the frame.
            get_img (bool, optional): whether to get the image. Defaults to True.
            get_obj (bool, optional): whether to get the object. Defaults to True.

        Returns:
            Dict: the data of the frame.
        """
        index_chain = self.index_chains[data_idx]
        subject_id, sequence_name, view_idx, frame_idx = index_chain.split('.') # string
        image_name = osp.join(self.image_dir, index_chain.replace('.', '/') + '.jpg')
        anno_name  = osp.join(self.anno_dir,  index_chain.replace('.', '/') + '.pkl')

        image = cv2.cvtColor(cv2.imread(image_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) \
            if get_img else None

        anno = pickle.load(open(anno_name, 'rb'))

        cameraK = anno['cameraK'].reshape(3, 3)
        distort_coeffs = None if anno['distort_coeffs'] is None else anno['distort_coeffs'].reshape(8)

        mano_rvc_r, mano_rvc_l = anno['mano_pose_r'].reshape(16, 3), anno['mano_pose_l'].reshape(16, 3)
        mano_shape_r, mano_shape_l = anno['mano_shape_r'].reshape(10), anno['mano_shape_l'].reshape(10)
        mano_trans_r, mano_trans_l = anno['mano_trans_r'].reshape(3),  anno['mano_trans_l'].reshape(3)

        j2d_r, j2d_l = anno['j2d_r'].reshape(21, 2), anno['j2d_l'].reshape(21, 2)
        j3d_r, j3d_l = anno['j3d_r'].reshape(21, 3), anno['j3d_l'].reshape(21, 3)
        jvis_r, jvis_l = anno['jvis_r'].reshape(21), anno['jvis_l'].reshape(21)
        jvis_r, jvis_l = jvis_r[1:].reshape(5, 4), jvis_l[1:].reshape(5, 4) # no root joint, 5 fingers
        joint_valid_r, joint_valid_l = anno['valid_r'].reshape(21), anno['valid_l'].reshape(21)
        hand_valid_r,  hand_valid_l  = anno['hand_valid_r'], anno['hand_valid_l']

        bbox_r, bbox_l = produce_points2d_bbox(j2d_r, expansion=1.5), produce_points2d_bbox(j2d_l, expansion=1.5)

        # ````` generate multi-hypotheses for hand pose
        multi_pose_r, multi_pose_l = anno['multi_finger_pose_r'], anno['multi_finger_pose_l']
        hand_multi_rvc_r = np.zeros((self.num_hypotheses, 5, 3, 3), dtype=np.float32)
        hand_multi_rvc_l = np.zeros((self.num_hypotheses, 5, 3, 3), dtype=np.float32)
        for finger_idx in range(5): # 5 fingers
            finger_multi_pose_r = multi_pose_r[finger_idx].reshape(-1, 3, 3)
            finger_multi_pose_l = multi_pose_l[finger_idx].reshape(-1, 3, 3)
            rand_ind_r = np.random.randint(0, len(finger_multi_pose_r), self.num_hypotheses)
            rand_ind_l = np.random.randint(0, len(finger_multi_pose_l), self.num_hypotheses)
            hand_multi_rvc_r[:, finger_idx] = finger_multi_pose_r[rand_ind_r]
            hand_multi_rvc_l[:, finger_idx] = finger_multi_pose_l[rand_ind_l]
        hand_multi_rvc_r = hand_multi_rvc_r.reshape(self.num_hypotheses, 15, 3)
        hand_multi_rvc_l = hand_multi_rvc_l.reshape(self.num_hypotheses, 15, 3)

        if get_obj:
            obj_verts, obj_faces = self.get_artic_object(anno['obj_name'], anno['obj_top_world2cam_Rt'],
                                                         anno['obj_bottom_world2cam_Rt'])
        else:
            obj_verts, obj_faces = None, None

        frame_data = {}
        if get_img:
            frame_data.update(dict(image=image))
        if get_obj:
            frame_data.update(dict(obj_verts=obj_verts, obj_faces=obj_faces))

        frame_data.update(dict(
            cameraK=cameraK, distort_coeffs=distort_coeffs,

            mano_rvc_r=mano_rvc_r, mano_rvc_l=mano_rvc_l,
            mano_shape_r=mano_shape_r, mano_shape_l=mano_shape_l,
            mano_trans_r=mano_trans_r, mano_trans_l=mano_trans_l,
            hand_multi_rvc_r=hand_multi_rvc_r, hand_multi_rvc_l=hand_multi_rvc_l,

            j2d_r=j2d_r, j2d_l=j2d_l,
            j3d_r=j3d_r, j3d_l=j3d_l,

            jvis_r=jvis_r, jvis_l=jvis_l,
            joint_valid_r=joint_valid_r, joint_valid_l=joint_valid_l,
            hand_valid_r=hand_valid_r, hand_valid_l=hand_valid_l,

            bbox_r=bbox_r, bbox_l=bbox_l,
            subject_id=subject_id,
        ))

        return frame_data

    # ------------------------------
    def get_artic_object(self, obj_name:str, top_world2cam_Rt:np.ndarray, bottom_world2cam_Rt:np.ndarray):
        """
        Combine two parts of articulated objects into a complete object.

        Args:
            obj_name (str): the name of the object.
            top_world2cam_Rt (np.ndarray): the transformation matrix from the top part of the object to the camera.
            bottom_world2cam_Rt (np.ndarray): the transformation matrix from the bottom part of the object to the camera.
        """

        obj_meshes = deepcopy(self.objects[obj_name])

        obj_verts = obj_meshes['verts']
        is_top = obj_meshes['is_top']
        obj_verts[ is_top] = (_padding(obj_verts[ is_top]) @ top_world2cam_Rt.T)[:, :3]
        obj_verts[~is_top] = (_padding(obj_verts[~is_top]) @ bottom_world2cam_Rt.T)[:, :3]
        obj_faces = obj_meshes['faces']

        return obj_verts, obj_faces

    # ------------------------------
    @staticmethod
    def distort_pts(pts:np.ndarray, coeffs:np.ndarray):
        """
        For ego camera ('p2' protocal), distort the 3D points in ego camera space before projection.
        """
        assert pts.ndim == 2 and pts.shape[1] == 3, 'pts should be Nx3-dim.'
        assert len(coeffs) == 8, 'coefficients should be 8-dim.'
        eps = 1e-8

        z = pts[..., 2]
        x1, y1 = pts[..., 0] / (z + eps), pts[..., 1] / (z + eps)
        x1_2, y1_2, x1_y1 = x1 * x1, y1 * y1, x1 * y1

        r2 = x1_2 + y1_2
        r4 = r2 * r2
        r6 = r4 * r2

        r_dist = (1 + coeffs[0] * r2 + coeffs[1] * r4 + coeffs[4] * r6) \
               / (1 + coeffs[5] * r2 + coeffs[6] * r4 + coeffs[7] * r6)

        # full (rational + tangential) distortion
        x2 = x1 * r_dist + 2 * coeffs[2] * x1_y1 + coeffs[3] * (r2 + 2 * x1_2)
        y2 = y1 * r_dist + 2 * coeffs[3] * x1_y1 + coeffs[2] * (r2 + 2 * y1_2)

        # denormalize for projection (which is a linear operation)
        distored_pts = np.stack([x2 * z, y2 * z, z], axis=-1)

        return distored_pts

    # ------------------------------
    def decimate_frames(self, decimation:int=None):
        """train on a subset of the frames for faster training. Perserve the frames every 'deciamation' frames.
        """
        if decimation is None:
            return self.index_chains

        assert decimation > 0, 'Decimation interval should be greater than 0.'
        if self.protocal == 'p1': # 8 views
            num_frames = len(self.index_chains) // 8
            view_indices = np.arange(0, 8, 1)
            dec_frame_indices = np.arange(0, num_frames, decimation)
            dec_indices = view_indices.reshape(1, -1) + dec_frame_indices.reshape(-1, 1) * 8
            new_index_chains = [self.index_chains[i] for i in dec_indices.reshape(-1)]
        elif self.protocal == 'p2': # only one view
            new_index_chains = self.index_chains[::decimation]
        else:
            ...

        return new_index_chains



# ------------------------------
if __name__ == '__main__':
    dataset = ArcticFrameDataset(data_dir='/root/Workspace/DATASETS/ARCTIC', mode='train', resolution=256,
                                 hand_type='right', protocal='p1', num_hypotheses=10, decimation=4)
    print(len(dataset))
    data = dataset[0]
