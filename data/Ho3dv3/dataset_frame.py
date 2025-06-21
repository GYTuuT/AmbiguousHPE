
import os
import sys

sys.path.append(os.getcwd())

import os.path as osp
import pickle
import random
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import Dataset

from tools.dataTransform import (AugmentPipe, apply_crop_resize, rmt_2_rvc,
                                 rvc_2_rmt, produce_points2d_bbox)


# ----------------
class Ho3dv3FrameDataset(Dataset):

    def __init__(self,
                 data_dir:str,
                 mode:str,
                 resolution:int=256,
                 num_hypotheses: int=100,
                 proportion_hypotheses:int=1.0, # proportion of ambiguous hypotheses
                 **kwargs) -> None:
        super().__init__()

        assert mode in ['train']

        self.data_dir = data_dir
        self.mode = mode
        self.resolution = resolution
        self.num_hypotheses = num_hypotheses
        self.proportion_hypotheses = proportion_hypotheses

        self.img_dir = osp.join(data_dir, mode)
        self.ann_dir = osp.join(data_dir, 'aranged_annotations')

        with open(osp.join(data_dir, f'{mode}.txt')) as txtfile:
            _content = txtfile.readlines()
        self.filelist = [line.strip() for line in _content]

        Jvis_Focc_info_name = osp.join(self.ann_dir, f'Ho3dv3_JvisFoccInfo_{mode}.pkl')
        self.Jvis_Focc_info = pickle.load(open(Jvis_Focc_info_name, 'rb'))

        finger_amb_rotvec_name = osp.join(self.ann_dir, f'Ho3dv3_FingerAmbRotvec_{mode}.pkl')
        self.finger_amb_rotvec = pickle.load(open(finger_amb_rotvec_name, 'rb'))

        ycb_models_name = osp.join(self.data_dir, 'aranged_annotations', 'YCB_models_tiny.pkl')
        self.ycb_models = pickle.load(open(ycb_models_name, 'rb'))

        self.augment_pipe = AugmentPipe(color_prob=0.5, noise_prob=0.25, geometry_prob=0.75)


    # ----------
    def __len__(self):
        return len(self.filelist)

    # ----------------
    def __getitem__(self, index:int) -> Dict: # NOTE the output is in meters.
        dtype = torch.float32

        frame_data = self.fetch_frame_data(index, get_img=True, get_obj=True)
        frame_data = self.apply_transform(frame_data, do_augment=(self.mode=='train'))

        mano_rmt = rvc_2_rmt(torch.from_numpy(frame_data['mano_rvc'])).to(dtype=dtype) # [16, 3, 3]

        # process multi hypotheses
        multi_rvc = np.expand_dims(frame_data['mano_rvc'], axis=0).repeat(self.num_hypotheses, axis=0) # [N, 16, 3]
        if self.proportion_hypotheses > 0.0:
            num_replace = int(self.num_hypotheses * self.proportion_hypotheses) # number of hypotheses using ambigious anno.
            multi_rvc[:num_replace, 1:] = frame_data['hand_multi_rvc'][:num_replace] # random select N_hypo
            np.random.shuffle(multi_rvc) # shuffle the hypotheses
        multi_rvc = multi_rvc + np.random.randn(*multi_rvc.shape) * 0.01 # add noise to hypotheses
        multi_rmt = rvc_2_rmt(torch.from_numpy(multi_rvc).to(dtype=dtype)).reshape(self.num_hypotheses, 16, 3, 3)

        hand_trans = torch.from_numpy(frame_data['mano_trans']).to(dtype=dtype) # [3,]

        outputs = dict(
            is_right = torch.from_numpy(frame_data['is_right']).to(dtype=torch.bool),
            image = torch.from_numpy(frame_data['img'] / 255.).permute(2, 0, 1).to(dtype=dtype), # [3, H, W]

            root_rmt = mano_rmt[:1], # [1, 3, 3]
            hand_rmt = mano_rmt[1:], # [15, 3, 3]
            multi_rmt = multi_rmt[:, 1:], # [N, 15, 3, 3]

            hand_shape = torch.from_numpy(frame_data['mano_shape']).to(dtype=dtype), # [10,]
            hand_trans = hand_trans, # [3,]

            joint3d = torch.from_numpy(frame_data['joint3d']).to(dtype=dtype), # [21, 3]
            joint2d = torch.from_numpy(frame_data['joint2d']).to(dtype=dtype), # [21, 2]
            joint_vis = torch.from_numpy(frame_data['joint_vis']).to(dtype=torch.bool), # [5, 4],
            joint_valid = torch.ones(21, dtype=torch.bool), # all joints are valid for dexycb

            proj_K = torch.from_numpy(frame_data['proj_K']).to(dtype=dtype), # [3, 3]
        )

        return outputs


    # ----------------
    def apply_transform(self, frame_data:Dict, do_augment:bool=False, do_flip_left:bool=False) -> Dict:
        """Apply the transformation to the frame data.
        """
        # NOTE Once do augmentation and flip, the object information will be invalid.
        # NOTE Geometry augmentation will make the global transformation and projection matrix invalid.

        # 1) crop and resize the image.
        assert 'img' in frame_data, 'Image is not provided.'
        assert 'bbox2d' in frame_data, '2D bbox is not provided.'
        crop_image, crop_matrix = apply_crop_resize(frame_data['img'], frame_data['bbox2d'], (self.resolution,)*2)
        frame_data['img'] = crop_image
        frame_data['joint2d'] = (crop_matrix[:2, :2] @ frame_data['joint2d'].T + crop_matrix[:2, 2:]).T
        frame_data['proj_K'] = crop_matrix @ frame_data['camera_K'].copy() # [3, 3]

        # 2) augmentaion
        if do_augment: # NOTE rotation will break the standard projection matrix.
            assert hasattr(self, 'augment_pipe'), 'Augment pipe is not initialized.'
            augment_params = self.augment_pipe.get_augment_params_randomly()
            global_rotmat = rvc_2_rmt(frame_data['mano_rvc'][0].copy()) # [3, 3]
            aug_img, (aug_j2d, aug_j3d, aug_global_rotmat), aug_matrix = self.augment_pipe.forward_augment(
                augment_params, frame_data['img'], frame_data['joint2d'], frame_data['joint3d'], global_rotmat)
            frame_data['img'] = aug_img
            frame_data['joint2d'] = aug_j2d
            frame_data['joint3d'] = aug_j3d
            frame_data['mano_rvc'][0] = rmt_2_rvc(aug_global_rotmat)

        # # 3) do flip_left
        # if do_flip_left and (not frame_data['is_right']):
        #     frame_data['img'] = cv2.flip(frame_data['img'], 1)
        #     frame_data['bbox2d'][0] = self.resolution - frame_data['bbox2d'][0]
        #     frame_data['joint2d'][:, 0] = self.resolution - frame_data['joint2d'][:, 0]
        #     frame_data['joint3d'][:, 0] = - frame_data['joint3d'][:, 0] + 2 * frame_data['joint3d'][0, 0] # keep the same root.
        #     frame_data['mano_rvc'][..., 1:] = frame_data['mano_rvc'][..., 1:] * -1 # flip mano pose.
        #     frame_data['hand_multi_rvc'][..., 1:] = frame_data['hand_multi_rvc'][..., 1:] * -1

        return frame_data

    # ----------
    def fetch_frame_data(self, data_idx:int, get_img:bool=True, get_obj:bool=False):

        filename = self.filelist[data_idx]
        frame_seq, frame_idx = filename.split('/')

        if self.mode == 'test': # mode that only appears after custom split.
            ann_name = osp.join(self.ann_dir, 'train', f'{filename}.pkl')
        else:
            ann_name = osp.join(self.ann_dir, self.mode, f'{filename}.pkl')
        img_name = osp.join(self.img_dir, frame_seq, 'rgb', f'{frame_idx}.jpg')

        image = None
        if get_img:
            image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anno = pickle.load(open(ann_name, 'rb'))

        # 1. hand information
        mano_rvc = anno['handPose'].reshape(16, 3)
        mano_shape = anno['handBeta'].reshape(10)
        # mano_trans = anno['handTrans'].reshape(3)
        mano_trans = anno['handTrans_globalLast'].reshape(3)

        cameraK = anno['camMat'].astype(np.float32).reshape(3, 3)

        joint3d = anno['handJoints3D'].reshape(21, 3)
        joint2d = ((joint3d / (joint3d[:, 2:] + 1e-8)) @ cameraK.T)[:, :2]
        bbox = produce_points2d_bbox(joint2d, expansion=1.5)

        # 2. occlusion and ambiguity infomation
        joints_vis = self.Jvis_Focc_info[data_idx]['Jvis'] # [5, 4], visiable joint set True
        finger_occ = self.Jvis_Focc_info[data_idx]['Focc'] # [5,], occluede finger set True

        hand_multi_rvc = np.zeros([self.num_hypotheses, 5, 3, 3], dtype=np.float32) # only finger
        finger_amb = np.zeros([5]).astype(np.bool_)
        for finger_idx in range(5):
            ambrvs = self.finger_amb_rotvec[data_idx][finger_idx]
            finger_amb[finger_idx] = (len(ambrvs) >= 2) # the first is orignal pose, > 1 mean has ambiguity.
            rand_ind = np.random.randint(0, len(ambrvs), self.num_hypotheses) # randomly select finger pose.
            hand_multi_rvc[:, finger_idx] = ambrvs[rand_ind]
        hand_multi_rvc = hand_multi_rvc.reshape(self.num_hypotheses, 15, 3) # [N, 15, 3]

        # 3. object information, in ho3dv3, there only one object in a scene.
        if get_obj:
            obj_verts = self.ycb_models[anno['objName']]['verts']
            obj_faces = self.ycb_models[anno['objName']]['faces']
            obj_rotW2C = cv2.Rodrigues(anno['objRot'])[0]
            obj_verts = (obj_rotW2C @ obj_verts.T).T + anno['objTrans'][None]

        # 5. outputs
        out = dict()

        if get_img:
            out.update(dict(img=image))

        if get_obj:
            out.update(dict(obj_verts=obj_verts.reshape(2010, 3),
                            obj_faces=obj_faces.reshape(4000, 3)))

        out.update(dict(
            is_right = np.array([1.]).astype(int),

            mano_rvc = mano_rvc, # [16, 3]
            mano_shape = mano_shape, # [10]
            mano_trans = mano_trans, # [3]
            hand_multi_rvc = hand_multi_rvc, # [N, 15, 3]

            joint3d = joint3d, # [21, 3]
            joint2d = joint2d, # [21, 2]
            bbox2d = bbox, # [4,], [x1, y1, length, length]
            joint_vis = joints_vis, # [5, 4]

            camera_K = cameraK,
        ))

        return out




## ===================
class Ho3dv3MheFrameDataset(Ho3dv3FrameDataset):
    def __init__(self,
                 data_dir: str,
                 mode: str,
                 resolution: int = 256,
                 num_hypotheses: int = 100,
                 **kwargs) -> None:

        super().__init__(data_dir=data_dir,
                         mode='train',
                         resolution=resolution,
                         num_hypotheses=num_hypotheses,
                         **kwargs)

        assert mode in ['train', 'test']

        self.mode = mode
        train_ind, test_ind = self.MHE_split_train_test()
        select_ind = train_ind if mode == 'train' else test_ind

        self.filelist = [self.filelist[idx] for idx in select_ind]
        self.Jvis_Focc_info = [self.Jvis_Focc_info[idx] for idx in select_ind]
        self.finger_amb_rotvec = [self.finger_amb_rotvec[idx] for idx in select_ind]



    # ----------------
    def MHE_split_train_test(self):

        test_seq = ['ABF14', 'MC5', 'SB14', 'ShSu13'] # same as ICCV 2023 'MHEntropy: Entropy Meets Multiple Hypotheses for Pose and Shape Recovery'

        train_ind = []
        test_ind = []

        for idx, filename in enumerate(self.filelist):
            if filename.split('/')[0] in test_seq:
                test_ind.append(idx)
            else:
                train_ind.append(idx)

        return train_ind, test_ind





## -----------------
if __name__ == '__main__':
    dataset = Ho3dv3FrameDataset(data_dir='/root/Workspace/DATASETS/HO3D_v3', mode='train', resolution=256, num_hypotheses=100)
    data = dataset[0]




