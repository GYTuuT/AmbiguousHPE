
import os
import sys

sys.path.append(os.getcwd())

import os.path as osp
import pickle

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from main.modules.Manolayer import (Mano_Rotvec_Mean_Left,
                                    Mano_Rotvec_Mean_Right, ManoPcalayer,
                                    RotVectorManolayer)

_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]

_SERIALS = [ # represent different camera viewpoints.
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}



## ---------------
def get_dataset_allocation(config_id:str='s0', mode:str='train'):
    # config_id: ['s0','s1','s2','s3']
    # mode: ['train', 'val', 'test']

    subject_ind = []
    serial_ind = []
    sequence_ind = []

    # 1.Seen subjects, camera views, grasped objects.
    if config_id == 's0':
        if mode == 'train':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 != 4]
        if mode == 'val':
            subject_ind = [0, 1]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 == 4]
        if mode == 'test':
            subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 == 4]

    # 2.Unseen subjects.
    if config_id == 's1':
        if mode == 'train':
            subject_ind = [0, 1, 2, 3, 4, 5, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = list(range(100))
        if mode == 'val':
            subject_ind = [6]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = list(range(100))
        if mode == 'test':
            subject_ind = [7, 8]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = list(range(100))

    # 3.Unseen camera views.
    if config_id == 's2':
        if mode == 'train':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5]
            sequence_ind = list(range(100))
        if mode == 'val':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [6]
            sequence_ind = list(range(100))
        if mode == 'test':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [7]
            sequence_ind = list(range(100))

    # 4.Unseen grasped objects.
    if config_id == 's3':
        if mode == 'train':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [
                i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)
            ]
        if mode == 'val':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i // 5 in (3, 19)]
        if mode == 'test':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i // 5 in (7, 11, 15)]

    return subject_ind, serial_ind, sequence_ind



## -------------------
def generate_indexmap(data_dir:str='.', mode:str='train'):

    calibration_dir = osp.join(data_dir, 'calibration')

    chosen_subj_ind, chosen_serial_ind, chosen_seq_ind \
        = get_dataset_allocation('s0', mode)
    chosen_subject_names = [_SUBJECTS[idx] for idx in chosen_subj_ind]
    chosen_serial_names = [_SERIALS[idx] for idx in chosen_serial_ind]


    source_sequence_names = []
    source_frame_indexmap = [] # [[seq_id, serial_id, frame_id],...]
    source_hand_type_list = [] # relative to seq_id
    source_hand_beta_list = [] # relative to seq_id
    source_scene_ycb_names = [] # relative to seq_id
    source_grasp_ycb_ind  = [] # relative to seq_id, the index for scene_ycb_names
    source_cam_intrinsics = {} # relative to serial

    for subj_name in chosen_subject_names:

        subj_dir = osp.join(data_dir, subj_name)
        seq_names = sorted(os.listdir(subj_dir))
        assert len(seq_names) == 100, 'Each subject has 100 sequences.'
        chosen_seq_names = [seq_names[idx] for idx in chosen_seq_ind]
        chosen_seq_names = [osp.join(subj_name, name) for name in chosen_seq_names]

        for seq_idx, seq_name in enumerate(chosen_seq_names):
            meta_filename = osp.join(data_dir, seq_name, 'meta.yml')
            meta = yaml.load(open(meta_filename, 'r'), Loader=yaml.FullLoader)

            #``
            # 1.construct frame indexmap.
            n_serial_per_seq = len(chosen_serial_names)
            n_frames_per_ser = meta['num_frames']

            frame_indices, serial_indices = np.meshgrid(np.arange(n_frames_per_ser),
                                                        np.arange(n_serial_per_seq), indexing='xy')
            seq_idx_offset = len(source_sequence_names)
            seq_indices = (seq_idx_offset + seq_idx) * np.ones_like(frame_indices)
            seq_serial_frame_indexmap = np.stack((seq_indices.ravel(), serial_indices.ravel(), frame_indices.ravel()), axis=1)
            # indexmap is using as '[seq_id, serial_id, frame_id] = indexmap[0]'
            source_frame_indexmap.append(seq_serial_frame_indexmap)

            #``
            # 2.mano hand types and betas
            source_hand_type_list.append(meta['mano_sides'][0])

            calib_filename = osp.join(calibration_dir, 'mano_{}'.format(meta['mano_calib'][0]), 'mano.yml')
            calib = yaml.load(open(calib_filename, 'r'), Loader=yaml.FullLoader)
            source_hand_beta_list.append(np.array(calib['betas'], dtype=np.float32))

            #``
            # 3.appearing ycb models in the scene
            scene_ycb_names = [_YCB_CLASSES[i] for i in meta['ycb_ids']]
            source_scene_ycb_names.append(scene_ycb_names)
            source_grasp_ycb_ind.append(meta['ycb_grasp_ind'])

        source_sequence_names = source_sequence_names + chosen_seq_names
    source_frame_indexmap = np.concatenate(source_frame_indexmap, axis=0)

    #``
    # 4.camera intrinsics
    for serial_idx, serial_name in enumerate(_SERIALS):
        intri_name = osp.join(calibration_dir,
                              'intrinsics',
                              f'{serial_name}_640x480.yml')
        intri = yaml.load(open(intri_name, 'r'), Loader=yaml.FullLoader)['color']

        camera_k = np.eye(3)
        camera_k[0,0], camera_k[1,1] = intri['fx'], intri['fy']
        camera_k[0,2], camera_k[1,2] = intri['ppx'], intri['ppy']
        source_cam_intrinsics[serial_name] = camera_k

    #``
    # 5.clean hand-unseen data # NOTE
    preserved_ids = []
    for i in tqdm(range(len(source_frame_indexmap))):
        label_name = osp.join(data_dir,
                              source_sequence_names[source_frame_indexmap[i][0]],
                              _SERIALS[source_frame_indexmap[i][1]],
                              f'labels_{source_frame_indexmap[i][2]:06d}.npz')
        label = np.load(label_name)

        if np.all(label['joint_3d'] == -1):
            continue
        if np.all(label['joint_2d'] == -1):
            continue
        if np.all(label['pose_m'] == 0):
            continue

        preserved_ids.append(i)

    #``
    # 6.save aranged annotations
    aranged_anno = dict(
        frame_indexmap = source_frame_indexmap[preserved_ids],
        sequence_names = source_sequence_names,
        serial_names   = chosen_serial_names,

        hand_type_list = source_hand_type_list,
        hand_beta_list = source_hand_beta_list,

        scene_ycb_names = source_scene_ycb_names,
        grasp_ycb_ind  = source_grasp_ycb_ind ,

        cam_intrinsics = source_cam_intrinsics,
    )

    # print(f'There are {len(source_sequence_names)} Sequences with {len(_SERIALS )} Camera Views(Serials) and All {len(preserved_ids)} Frames in {mode} Data Annotations')

    # aranged_anno_name = osp.join(data_dir, 'custom_annotations', f'indexmap_{mode}.pkl')
    # os.makedirs(osp.dirname(aranged_anno_name), exist_ok=True)
    # pickle.dump(aranged_anno, open(aranged_anno_name, 'wb'))



## -----------------
def reprocess_image_labels(data_dir:str='.', mode:str='train'):

    annos_name = osp.join(data_dir, 'custom_annotations', f'indexmap_{mode}.pkl')
    annos = pickle.load(open(annos_name, 'rb'))
    seq_names = annos['sequence_names']
    ser_names = annos['serial_names']
    frame_indexmap = annos['frame_indexmap']

    Jvis_info_anno = pickle.load(open(osp.join(data_dir, f'Dexycb_JvisFoccInfo_{mode}.pkl'), 'rb'))
    Ambrvc_annos = pickle.load(open(osp.join(data_dir, f'Dexycb_FingerAmbRotvec_{mode}.pkl'), 'rb'))

    assert len(Jvis_info_anno) == len(frame_indexmap)
    assert len(Ambrvc_annos) == len(frame_indexmap)

    manolayers = {
        'right': RotVectorManolayer('checkpoints/processed_mano', hand_type='right', set_root_as_origin=False),
        'left': RotVectorManolayer('checkpoints/processed_mano', hand_type='left',
                                   set_root_as_origin=False, fix_left_shapedirs=False), # DexYCB do not consider the left shapedirs bug.
    }

    manolayers_ro = {
        'right': RotVectorManolayer('checkpoints/processed_mano', hand_type='right', set_root_as_origin=True),
        'left': RotVectorManolayer('checkpoints/processed_mano', hand_type='left',
                                   set_root_as_origin=True, fix_left_shapedirs=False),
    }

    r_pca_mat = ManoPcalayer('checkpoints/processed_mano').r_pca_mat.detach().numpy()
    l_pca_mat = ManoPcalayer('checkpoints/processed_mano').l_pca_mat.detach().numpy()

    r_pose_mean = np.array(Mano_Rotvec_Mean_Right) # [48]
    l_pose_mean = np.array(Mano_Rotvec_Mean_Left)

    image_format = 'color_{:06d}.jpg'
    label_format = 'labels_{:06d}.npz'
    new_label_format = 'labels_{:06d}.pkl'
    for idx in tqdm(range(len(frame_indexmap))):

        new_label = dict(hand=dict(
                            jvis=None, # [n_joints]
                            joint2d=None, # pixel
                            joint3d=None, # meter
                            mano_pose=None, # rotvec
                            mano_shape=None,
                            mano_trans=None,
                            amb_finger_pose=None,),
                         obj=dict( # objects can't be flipped in 3D space.
                            names=[],
                            grasp_id=None, # index of 'names'
                            rotation=[],
                            translation=[],),
                         cameraK=None,)

        seq_idx, ser_idx, frm_idx = frame_indexmap[idx]
        ser_dir = osp.join(data_dir, seq_names[seq_idx], ser_names[ser_idx])
        image_name = osp.join(ser_dir, image_format.format(frm_idx))
        label_name = osp.join(ser_dir, label_format.format(frm_idx))

        label = np.load(label_name)
        camK = annos['cam_intrinsics'][ser_names[ser_idx]].astype(np.float32)

        hand_type = annos['hand_type_list'][seq_idx]
        hand_beta = annos['hand_beta_list'][seq_idx]

        if hand_type == 'right':
            mano_rotvec = label['pose_m'][0, :48].astype(np.float32)
            mano_rotvec[3:] = mano_rotvec[3:] @ r_pca_mat
            mano_rotvec = mano_rotvec + r_pose_mean
        elif hand_type == 'left':
            mano_rotvec = label['pose_m'][0, :48].astype(np.float32)
            mano_rotvec[3:] = mano_rotvec[3:] @ l_pca_mat
            mano_rotvec = mano_rotvec + l_pose_mean
        else:
                raise KeyError

        joint3d = label['joint_3d'].astype(np.float32)[0]
        joint2d = label['joint_2d'].astype(np.float32)[0]

        mano_shape = hand_beta
        mano_trans = label['pose_m'][0, 48:].astype(np.float32)

        # replace the translation of mano.
        rvc_tensor = torch.from_numpy(mano_rotvec).reshape(1, 16, 3)
        shape_tensor = torch.from_numpy(mano_shape).reshape(1, 10)
        trans_tensor = torch.from_numpy(mano_trans).reshape(1, 3)

        ori_v3d, ori_j3d = manolayers[hand_type](rvc_tensor[:, :1], rvc_tensor[:, 1:], shape_tensor, trans_tensor)
        new_v3d, new_j3d = manolayers_ro[hand_type](rvc_tensor[:, :1], rvc_tensor[:, 1:], shape_tensor)
        ori_j3d, new_j3d = ori_j3d[0].numpy(), new_j3d[0].numpy()

        # print(hand_type)
        # print()
        # print(ori_j3d)
        # print()
        # print(joint3d)
        # print()
        # print(np.abs(ori_j3d - joint3d))
        # exit()

        if (np.abs(ori_j3d - joint3d) > 1e-2).any():
            raise ValueError(f'Mismatched 3D joint positions between Mano and Joint3D, idx {idx}')

        if (np.abs(ori_j3d - joint3d) > 1e-3).any():
            joint3d = ori_j3d.copy()

        new_trans = (ori_j3d - new_j3d).mean(axis=0).reshape(3)

        # attach the finger ambiguous rotation vector and jvis info

        new_label['hand']['jvis'] = Jvis_info_anno[idx]['Jvis'].copy()
        new_label['hand']['joint2d'] = joint2d.copy()
        new_label['hand']['joint3d'] = joint3d.copy()
        new_label['hand']['mano_pose'] = mano_rotvec.copy()
        new_label['hand']['mano_shape'] = mano_shape.copy()
        new_label['hand']['mano_trans'] = new_trans.copy()
        new_label['hand']['amb_finger_pose'] = Ambrvc_annos[idx].copy() # n_fingers * [n_hypo, n_j_per_finger, 3]

        new_label['obj']['names'] = annos['scene_ycb_names'][seq_idx]
        new_label['obj']['grasp_id'] = annos['grasp_ycb_ind'][seq_idx]
        new_label['obj']['rotation'] = label['pose_y'][..., :3]
        new_label['obj']['translation'] = label['pose_y'][..., 3:].transpose(0, 2, 1)

        new_label['cameraK'] = camK

        new_label_dir = osp.join(data_dir, 'custom_annotations',
                                 seq_names[seq_idx], ser_names[ser_idx])
        os.makedirs(new_label_dir, exist_ok=True)
        new_label_name = osp.join(new_label_dir, new_label_format.format(frm_idx))
        pickle.dump(new_label, open(new_label_name, 'wb'))





# ----------------------------------------------
if __name__ == '__main__':

    for mode in ['train', 'val', 'test']:
        generate_indexmap(data_dir='/root/Workspace/DATASETS/DexYCB', mode=mode)

    # for mode in ['train', 'test']:
    #     reprocess_image_labels(data_dir='/root/Workspace/DATASETS/DexYCB', mode=mode)




