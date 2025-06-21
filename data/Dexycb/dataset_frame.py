
import os
import sys

sys.path.append(os.getcwd())

import os.path as osp
import pickle
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import Dataset

from tools.dataTransform import (AugmentPipe, apply_crop_resize, flip2d,
                                 produce_points2d_bbox, rmt_2_rvc, rvc_2_rmt)


# ----------------
class DexycbFrameDataset(Dataset):
    """NOTE In Dexycb, the Left Hand Use Manolayer with Unfixed Shapedirs.
    """
    # ----------------
    def __init__(self,
                 data_dir:str,
                 mode:str,
                 resolution:int=256,
                 flip_left_hand:bool=True, # if flip left hand to right
                 num_hypotheses:int=100, # number of produced ambiguous hypotheses
                 proportion_hypotheses:int=0.0, # proportion of ambiguous hypotheses
                 decimation:int=None, # decimate the frames for faster training.
                 **kwargs) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.anno_dir = osp.join(data_dir, 'custom_annotations')
        self._image_format = 'color_{:06d}.jpg'
        self._label_format = 'labels_{:06d}.pkl'

        assert mode in ['train', 'test', 'val']
        self.mode = mode

        self.resolution = resolution
        self.flip_left_hand = flip_left_hand
        self.num_hypotheses = num_hypotheses
        self.proportion_hypotheses = proportion_hypotheses

        indexmap_name = osp.join(self.anno_dir, f'indexmap_{mode}.pkl')
        indexmap_anno = pickle.load(open(indexmap_name, 'rb'))
        self.frame_indexmap = indexmap_anno['frame_indexmap']
        self.sequence_names = indexmap_anno['sequence_names']
        self.serial_names   = indexmap_anno['serial_names'  ]
        self.hand_type_list = indexmap_anno['hand_type_list']

        self.frame_indexmap = self.frame_indexmap[::decimation] \
            if decimation is not None else self.frame_indexmap

        self.ycb_object_models = pickle.load(open(
            osp.join(self.anno_dir, 'YCB_models_tiny.pkl'), 'rb')) # 2010 verts, 4000 faces

        self.augment_pipe = AugmentPipe(color_prob=0.5, noise_prob=0.25, geometry_prob=0.5) # no geometry augmentation.

    # ----------------
    def __len__(self) -> int:
        return len(self.frame_indexmap)

    # ----------------
    def __getitem__(self, index:int) -> Dict: # NOTE the output is in meters.
        dtype = torch.float32

        frame_data = self.fetch_frame_data(index, get_img=True, get_obj=True)
        frame_data = self.apply_transform(frame_data, do_augment=(self.mode=='train'), do_flip_left=self.flip_left_hand)

        mano_rmt = rvc_2_rmt(torch.from_numpy(frame_data['mano_rvc'])).to(dtype=dtype) # [16, 3, 3]

        # process multi hypotheses
        multi_rvc = np.expand_dims(frame_data['mano_rvc'], axis=0).repeat(self.num_hypotheses, axis=0) # [N, 16, 3]
        if self.proportion_hypotheses > 0.0:
            num_replace = int(self.num_hypotheses * self.proportion_hypotheses) # number of hypotheses using ambigious anno.
            multi_rvc[:num_replace, 1:] = frame_data['hand_multi_rvc'][:num_replace] # random select N_hypo
            np.random.shuffle(multi_rvc) # shuffle the hypotheses
        multi_rvc = multi_rvc + np.random.randn(*multi_rvc.shape) * 0.01 # add noise to hypotheses
        multi_rmt = rvc_2_rmt(torch.from_numpy(multi_rvc).to(dtype=dtype)).reshape(self.num_hypotheses, 16, 3, 3)

        hand_trans = torch.from_numpy(frame_data['joint3d'][0]).to(dtype=dtype) # [3,]

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

        # 3) do flip_left
        if do_flip_left and (not frame_data['is_right']):
            frame_data['img'] = cv2.flip(frame_data['img'], 1)
            frame_data['bbox2d'][0] = self.resolution - frame_data['bbox2d'][0]
            frame_data['joint2d'][:, 0] = self.resolution - frame_data['joint2d'][:, 0]
            frame_data['joint3d'][:, 0] = - frame_data['joint3d'][:, 0] + 2 * frame_data['joint3d'][0, 0] # keep the same root.
            frame_data['mano_rvc'][..., 1:] = frame_data['mano_rvc'][..., 1:] * -1 # flip mano pose.
            frame_data['hand_multi_rvc'][..., 1:] = frame_data['hand_multi_rvc'][..., 1:] * -1

        return frame_data

    # ----------------
    def fetch_frame_data(self, data_idx:int, get_img:bool=True, get_obj:bool=True) -> Dict:
        """Fetch the data of a frame.

        Args:
            data_idx (int): the index of the frame.
            get_img (bool, optional): whether to get the image. Defaults to True.
            get_obj (bool, optional): whether to get the object. Defaults to True.

        Returns:
            Dict: the data of the frame.
        """
        seq_idx, viw_idx, frm_idx = self.frame_indexmap[data_idx]

        viw_name = osp.join(self.sequence_names[seq_idx], self.serial_names[viw_idx])
        img_name = osp.join(self.data_dir, viw_name, self._image_format.format(frm_idx))
        ann_name = osp.join(self.anno_dir, viw_name, self._label_format.format(frm_idx))
        # try:
        #     subject_name = re.search(r'subject-(\d{1,2})', viw_name).group(0)
        # except:
        #     print(viw_name)
        #     subject_name = 'unknown'

        img = cv2.cvtColor(cv2.imread(img_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) \
            if get_img else None

        ann = pickle.load(open(ann_name, 'rb'))
        is_right = np.array((self.hand_type_list[seq_idx] == 'right')).astype(np.bool_)

        mano_rvc   = ann['hand']['mano_pose'].reshape(16, 3)
        mano_shape = ann['hand']['mano_shape'].reshape(10)
        mano_trans = ann['hand']['mano_trans'].reshape(3)

        camera_K = ann['cameraK'].reshape(3, 3)
        joint3d = ann['hand']['joint3d'].reshape(21, 3)
        joint2d = (joint3d / joint3d[:, 2:3] @ camera_K.T)[:, :2].copy() # reproject the 3d joints to 2d.
        bbox2d = produce_points2d_bbox(joint2d, expansion=1.5)

        joint_visibilities = ann['hand']['jvis'].reshape(5, 4) # no root, true for visible
        hand_multi_poses = ann['hand']['amb_finger_pose'] # 5 x [N_hypo, 3, 3], 5 finger, each finger 3 rot joints, each joint 3 angles.

        hand_multi_rvc = np.zeros((self.num_hypotheses, 5, 3, 3), dtype=np.float32) # no root
        for fidx in range(5): # 5 fingers
            finger_multi_poses = hand_multi_poses[fidx] # [N_hypo, 3, 3]
            rand_ind = np.random.randint(0, len(finger_multi_poses), self.num_hypotheses) # random select N_hypo
            hand_multi_rvc[:, fidx] = finger_multi_poses[rand_ind]
        hand_multi_rvc = hand_multi_rvc.reshape(self.num_hypotheses, 15, 3)

        if get_obj: # may have multiple objects in one scene, stack them together
            obj_verts = [self.ycb_object_models[name]['verts'] for name in ann['obj']['names']]
            obj_faces = [self.ycb_object_models[name]['faces'] for name in ann['obj']['names']]
            obj_verts = np.stack(obj_verts, axis=0).astype(np.float32) # [N, 2010, 3]
            obj_faces = np.stack(obj_faces, axis=0).astype(np.int32) # [N, 4000, 3]
            obj_R, obj_T = ann['obj']['rotation'], ann['obj']['translation']
            obj_verts = obj_verts @ obj_R.transpose(0, 2, 1) + obj_T # to camera space
            verts_idx_offset = np.arange(0, len(obj_verts)) * obj_verts.shape[1] # reorder the faces.
            obj_faces = obj_faces + verts_idx_offset.reshape(-1, 1, 1)

        frame_data = {}
        if get_img:
            frame_data.update({'img': img})
        if get_obj:
            frame_data.update({'obj_verts': obj_verts.reshape(len(obj_verts) * 2010, 3),
                               'obj_faces': obj_faces.reshape(len(obj_verts) * 4000, 3)})

        frame_data.update(dict(
            is_right = is_right, # [1,]

            mano_rvc = mano_rvc, # [16, 3]
            mano_shape = mano_shape, # [10,]
            mano_trans = mano_trans, # [3,]
            hand_multi_rvc = hand_multi_rvc, # [N, 15, 3]

            joint3d = joint3d, # [21, 3]
            joint2d = joint2d, # [21, 2]
            bbox2d = bbox2d, # [4,], [x1, y1, length, length]
            joint_vis = joint_visibilities, # [5, 4]

            camera_K = camera_K, # [3, 3]
            # subject_name = subject_name, # [1,]
        ))

        return frame_data

    # ----------------
    @property
    def sequence_frame_indices(self) -> List[ndarray]:
        """
        Return the frame indices of each sequence. List of 1-d arrays, each array is the frame indices of a sequence.
        """
        seq_list = []
        cur_sidx, cur_vidx, cur_fidx = self.frame_indexmap[0]
        cur_seq = [0]
        for data_idx in range(1, len(self.frame_indexmap)):
            sidx, vidx, fidx = self.frame_indexmap[data_idx]
            if (sidx == cur_sidx) and (vidx == cur_vidx): # belong to same sequence.
                cur_seq.append(data_idx)
            else: # begin a new sequence.
                seq_list.append(np.array(cur_seq, dtype=np.int32))
                cur_seq = [data_idx]
                cur_sidx, cur_vidx = sidx, vidx
        seq_list.append(np.array(cur_seq, dtype=np.int32)) # the last sequence.

        return seq_list






if __name__ == '__main__':


    import random
    from tools.browserPloty import BrowserPlot
    import time

    dataset = DexycbFrameDataset('/root/Workspace/DATASETS/DexYCB', 'train')
    data_idx = random.randint(0, len(dataset)-1)
    frame_data = dataset[data_idx]

    t1 = time.time()
    for _ in range(50):
        data_idx = random.randint(0, len(dataset)-1)
        frame_data = dataset[data_idx]

        j2d = frame_data['joint2d'].reshape(21, 2).numpy()
        j3d = frame_data['joint3d'].reshape(21, 3).numpy()
        image = frame_data['image'].permute(1, 2, 0).cpu().numpy()

        brp = BrowserPlot(2, 1)
        brp.image(image, [1, 1], 'image')
        brp.scatter2D(j2d, [1, 1], 'joint2d', color='red')
        brp.scatter3D(j3d, [2, 1], 'joint3d', color='blue')
        brp.show()

        exit()

