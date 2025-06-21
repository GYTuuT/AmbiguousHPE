# import os
# import sys

# sys.path.append(os.getcwd())

import pickle
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from tools.metrics import compute_Joints_AUC, get_PaEuc_batch
from .dataset_frame import DexycbFrameDataset


Dataset = DexycbFrameDataset(data_dir='/root/Workspace/DATASETS/DexYCB', mode='test', flip_left_hand=True)

"""
1. Evaluate Root-Align MPJPE:
"""
def eval_Dexycb_Mpjpe(pred_joints:List[np.ndarray], flip_left:bool=True):
    """
    pred_joints: [78768, 21, 3], in meters, notice the data order and joint order
    flip_left: whether flip pred joints back to left hand.
    """
    global Dataset

    assert len(pred_joints) == len(Dataset)

    Mpjpe = np.zeros([len(pred_joints)], dtype=np.float32)
    Mpjpe_right_hand = []
    Mpjpe_left_hand = []

    print('Evaluating ...')
    ra_Js_gt = np.zeros([len(pred_joints), 21, 3])
    ra_Js_pr = np.zeros([len(pred_joints), 21, 3])
    for idx in tqdm(range(len(pred_joints))):

        gts = Dataset.fetch_frame_data(idx, get_img=False, get_obj=False)

        J_gt = gts['joint3d'] * 1000 # to mm
        J_pr = pred_joints[idx] * 1000

        if flip_left and (not gts['is_right'].item()): # flip back to left hand.
            J_pr[..., 0] *= -1

        # root align
        J_gt = (J_gt - J_gt[:1, ...])
        J_pr = (J_pr - J_pr[:1, ...])
        ra_Js_gt[idx] = J_gt
        ra_Js_pr[idx] = J_pr

        Mpjpe[idx] = (np.sqrt(np.square(J_pr - J_gt).sum(-1))).mean()

        if gts['is_right'].item():
            Mpjpe_right_hand.append(Mpjpe[idx])
        else:
            Mpjpe_left_hand.append(Mpjpe[idx])

    print(f'RA-MPJPE of hand joints: {Mpjpe.mean(): 0.3f}')
    print(f'RA-MPJPE of right hand joints: {np.mean(Mpjpe_right_hand): 0.3f}')
    print(f'RA-MPJPE of left hand joints: {np.mean(Mpjpe_left_hand): 0.3f}')
    PaMpjpe = get_PaEuc_batch(ra_Js_pr, ra_Js_gt)
    print(f'PA-MPJPE of hand joints: {PaMpjpe: 0.3f}')


"""
2. Evaluate Occluded finger and No Occluded finger Root-Align MPJPE:
"""
def eval_Dexycb_FingerOccluded_Mpjpe(pred_joints:List[np.ndarray], flip_left:bool=True):
    """
    pred_joints: [78768, 21, 3], in meters, notice the data order and joint order
    flip_left: whether flip pred joints back to left hand.
    """

    global Dataset

    assert len(pred_joints) == len(Dataset)

    N_occ = 0
    Err_occ_ra = 0
    Err_occ_pa = 0

    N_no_occ = 0
    Err_no_occ_ra = 0
    Err_no_occ_pa = 0

    print('Evaluating ...')
    for idx in tqdm(range(len(Dataset))):

        gts = Dataset.fetch_frame_data(idx, get_img=False, get_obj=False)
        F_occ = np.logical_not(gts['joint_vis']).any(-1) # if any joint is not visible, then the finger is occluded.
        J_occ = np.repeat(F_occ[..., None], axis=-1, repeats=4) # [5, 4]

        J_gt = gts['joint3d'] * 1000 # to mm
        J_pr = pred_joints[idx] * 1000

        if flip_left and (not gts['is_right'].item()): # flip back to left hand.
            J_pr[..., 0] *= -1

        # root align
        J_gt = (J_gt - J_gt[:1, ...])
        J_pr = (J_pr - J_pr[:1, ...])
        J_pr_aligned, _ = get_PaEuc_batch(J_pr[None], J_gt[None], return_aligned=True)
        J_pr_aligned = J_pr_aligned[0] # [21, 3]

        # reshape to finger shape
        J_gt = J_gt[1:].reshape(5, 4, 3)
        J_pr = J_pr[1:].reshape(5, 4, 3)
        J_pr_aligned = J_pr_aligned[1:].reshape(5, 4, 3)

        Err_occ_ra += (np.sqrt(np.square(J_pr - J_gt).sum(-1)) * J_occ).sum()
        Err_occ_pa += (np.sqrt(np.square(J_pr_aligned - J_gt).sum(-1)) * J_occ).sum()
        N_occ += J_occ.sum()

        Err_no_occ_ra += (np.sqrt(np.square(J_pr - J_gt).sum(-1)) * (~ J_occ)).sum()
        Err_no_occ_pa += (np.sqrt(np.square(J_pr_aligned - J_gt).sum(-1)) * (~ J_occ)).sum()
        N_no_occ += (~ J_occ).sum()


    occ_RaMpjpe = Err_occ_ra / (N_occ + 1e-8)
    no_occ_RaMpjpe = Err_no_occ_ra / (N_no_occ + 1e-8)

    occ_PaMpjpe = Err_occ_pa / (N_occ + 1e-8)
    no_occ_PaMpjpe = Err_no_occ_pa / (N_no_occ + 1e-8)

    print(f'RA-MPJPE of Occluded Fingers: {occ_RaMpjpe :0.3f}')
    print(f'RA-MPJPE of Not Occluded Figners: {no_occ_RaMpjpe :0.3f}')

    print(f'PA-MPJPE of Occluded Fingers: {occ_PaMpjpe :0.3f}')
    print(f'PA-MPJPE of Not Occluded Figners: {no_occ_PaMpjpe :0.3f}')

    return occ_RaMpjpe, no_occ_RaMpjpe


"""
6. Evaluate Joints3D AUC:
"""
def eval_Dexycb_Joints_AUC(pred_joints:List[np.ndarray], flip_left:bool=True):
    """
    pred_joints: [78768, 21, 3], in meters, notice the data order and joint order
    flip_left: whether flip pred joints back to left hand.
    """

    global Dataset
    assert len(pred_joints) == len(Dataset)

    pred_joints = [j3d * 1000 for j3d in pred_joints]
    gt_joints = []
    for idx in tqdm(range(len(Dataset))):

        gts = Dataset.fetch_frame_data(idx, get_img=False, get_obj=False)
        gt_j3d = gts['joint3d'] * 1000

        if flip_left and (not gts['is_right'].item()):
            gt_j3d[..., 0] *= -1

        gt_joints.append(gt_j3d)

    auc = compute_Joints_AUC(pred_joints, gt_joints)
    print(f'Joints AUC: {auc: 0.3f}')



