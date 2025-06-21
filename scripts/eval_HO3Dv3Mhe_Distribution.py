import os
import sys

sys.path.append(os.getcwd())

import pickle

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.Ho3dv3.dataset_frame import Ho3dv3MheFrameDataset
from main.model import AmbHand
from tools.p3dRender import batch_project_K
from tools.metrics import get_PaEuc_batch


#
# ----------
device = torch.device('cuda', 0)
torch.manual_seed(0)


# ----------
def fn_BH_RaMpjpe(preds:Tensor, gts:Tensor):
    """
    preds: [B, N, 21, 3]
    gts: [B, 21, 3]
    """
    ra_preds, ra_gts = preds - preds[..., :1, :], gts - gts[..., :1, :]
    ra_mpjpe = ((ra_preds - ra_gts.unsqueeze(1)).square().sum(-1) + 1e-8).sqrt().mean(-1) # [B, N]
    best = torch.topk(ra_mpjpe, k=1, dim=1, largest=False).values.squeeze(-1) # [B]

    return best # [B]


# ----------
def fn_BH_RaMpvpe(preds:Tensor, pred_roots:Tensor,
                  gts:Tensor, gt_roots:Tensor):
    """
    preds: [B, N, 778, 3]
    pred_roots: [B, N, 3]
    gts: [B, 778, 3]
    gt_roots: [B, 3]
    """
    ra_preds = preds - pred_roots.unsqueeze(-2)
    ra_gts = gts - gt_roots.unsqueeze(-2)
    ra_mpvpe = ((ra_preds - ra_gts.unsqueeze(1)).square().sum(-1) + 1e-8).sqrt().mean(-1) # [B, N]
    best = torch.topk(ra_mpvpe, k=1, dim=1, largest=False).values.squeeze(-1) # [B]

    return best # [B]


# -----------
def fn_VisJ2d(preds:Tensor, gts:Tensor, vis:Tensor): # i.e. the 'AH' metric.
    """
    preds: [B, N, 21, 2]
    gts: [B, 21, 2]
    vis: [B, 21]
    """
    err = ((preds - gts.unsqueeze(1)).square().sum(-1) + 1e-8).sqrt().mean(1) # [B, 21]
    err = (err * vis).sum(-1) / (vis.sum(-1) + 1e-8) # [B]

    return err # [B]


# -----------
def fn_VisStd_OccStd(pred_j2ds:Tensor, pred_j3ds:Tensor, vis:Tensor): # i.e. the 'PJD' metric.
    """
    pred_j2ds: [B, N, 21, 2]
    pred_j3ds: [B, N, 21, 3]
    vis: [B, 21]
    """
    j2d_center = pred_j2ds.mean(1, keepdim=True) # [B, 1, 21, 2]
    j3d_center = pred_j3ds.mean(1, keepdim=True) # [B, 1, 21, 3]

    vis_std = ((pred_j2ds - j2d_center).square().sum(-1) + 1e-8).sqrt().mean(1) # [B, 21]
    occ_std = ((pred_j3ds - j3d_center).square().sum(-1) + 1e-8).sqrt().mean(1) # [B, 21]

    occ = vis.logical_not()
    vis_std = (vis_std * vis).sum(-1) / (vis.sum(-1) + 1e-8) # [B]
    occ_std = (occ_std * occ).sum(-1) / (occ.sum(-1) + 1e-8) # [B]

    return vis_std, occ_std # [B]





# ---------------
@torch.no_grad()
def eval_HO3Dv3_MHE(model:AmbHand, stage:str='S2', resolution:int=256):

    assert stage in ['S1', 'S2']

    # 1. initialize
    model = model.to(device)
    model.eval()

    testset = Ho3dv3MheFrameDataset(data_dir='/root/Workspace/DATASETS/HO3D_v3', mode='test', resolution=resolution)
    testloader = DataLoader(testset, batch_size=64, num_workers=6, drop_last=False, shuffle=False)
    testJvis =  torch.from_numpy(
        pickle.load(open('checkpoints/MHEntropy_datas/Ho3dv3_MHE_Jvis.pkl', 'rb'))['test']
        ).to(device=device, dtype=torch.bool) # use MHEntropy's config.

    manolayer = model.stage_1_modules.rManolayer


    # 2. evaluate
    MPJPEs = torch.zeros([len(testset)], dtype=torch.float32, device=device)
    MPVPEs = torch.zeros([len(testset)], dtype=torch.float32, device=device)
    Bh_MPJPEs = torch.zeros([len(testset)], dtype=torch.float32, device=device)
    Bh_MPVPEs = torch.zeros([len(testset)], dtype=torch.float32, device=device)
    Ahs = torch.zeros([len(testset)], dtype=torch.float32, device=device)
    Pjd_2Ds = torch.zeros([len(testset)], dtype=torch.float32, device=device)
    Pjd_3Ds = torch.zeros([len(testset)], dtype=torch.float32, device=device)

    counter = 0
    for batch_data in tqdm(testloader):
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(device=device)

        # 2.0 Get predictions
        with torch.no_grad():
            S1_preds, S2_preds = model(batch_data, only_stage_1=True if stage=='S1' else False)

        if stage == 'S1':
            pd_amb_rmt = S1_preds['amb_rmt']
            pd_amb_shape = S1_preds['amb_shape']
        else:
            pd_amb_rmt = S2_preds['amb_rmt']
            pd_amb_shape = S2_preds['amb_shape']

        B, N = pd_amb_rmt.shape[:2]

        # 2.1 Get GTs
        gt_j2d = batch_data['joint2d']
        gt_Jvis = testJvis[counter:counter+B]

        gt_rmt = torch.cat([batch_data['root_rmt'], batch_data['hand_rmt']], dim=1) # [B, 16, 3, 3]
        gt_shape = batch_data['hand_shape'] # [B, 10]
        gt_trans = batch_data['hand_trans'] # [B, 3]
        gt_projK = batch_data['proj_K'] # [B, 3, 3]


        # 2.2 Process predictions
        pd_amb_rmt = pd_amb_rmt.reshape(B*N, 16, 3, 3)
        pd_amb_shape = pd_amb_shape.reshape(B*N, 10)
        pd_amb_v3d, pd_amb_j3d = manolayer(pd_amb_rmt[:, :1], pd_amb_rmt[:, 1:], pd_amb_shape,
                                           gt_trans.unsqueeze(1).repeat(1, N, 1).reshape(-1, 3)) # [B*N, 778, 3], [B*N, 21, 3]
        pd_amb_j2d = (pd_amb_j3d / (pd_amb_j3d[..., 2:] + 1e-8)) \
                   @ (gt_projK.transpose(1, 2).unsqueeze(1).expand(-1, N, -1, -1)).reshape(-1, 3, 3) # [B*N, 21, 3]
        pd_amb_j2d = pd_amb_j2d[..., :2]

        gt_v3d, gt_j3d = manolayer(gt_rmt[:, :1], gt_rmt[:, 1:], gt_shape, gt_trans) # [B, 778, 3], [B, 21, 3]

        # 2.3 Compute metrics
        pd_amb_j2d = pd_amb_j2d.reshape(B, N, 21, 2)
        pd_amb_j3d = pd_amb_j3d.reshape(B, N, 21, 3) * 1000
        pd_amb_v3d = pd_amb_v3d.reshape(B, N, 778, 3) * 1000

        gt_j3d = gt_j3d.reshape(B, 21, 3) * 1000
        gt_v3d = gt_v3d.reshape(B, 778, 3) * 1000

        # MPJPEs[counter:counter+B] = ((((pd_amb_j3d[:,0] - pd_amb_j3d[:,0][...,:1,:]) - (gt_j3d - gt_j3d[..., :1, :])
        #                               ).square().sum(-1) + 1e-8
        #                              ).sqrt() * gt_Jvis).sum(-1) / (gt_Jvis.sum(-1) + 1e-8)
        MPJPEs[counter:counter+B] = (((pd_amb_j3d[:,0] - pd_amb_j3d[:,0][...,:1,:]) - (gt_j3d - gt_j3d[..., :1, :])
                                      ).square().sum(-1) + 1e-8
                                     ).sqrt().mean()
        MPVPEs[counter:counter+B] = (((pd_amb_v3d[:,0] - pd_amb_j3d[:,0][...,:1,:]) - (gt_v3d - gt_j3d[..., :1, :])
                                     ).square().sum(-1) + 1e-8
                                    ).sqrt().mean(-1)

        Bh_MPJPEs[counter:counter+B] = fn_BH_RaMpjpe(pd_amb_j3d, gt_j3d)
        Bh_MPVPEs[counter:counter+B] = fn_BH_RaMpvpe(pd_amb_v3d, pd_amb_j3d[..., 0, :],
                                                      gt_v3d, gt_j3d[..., 0, :])
        Ahs[counter:counter+B] = fn_VisJ2d(pd_amb_j2d, gt_j2d, gt_Jvis)
        Pjd_2Ds[counter:counter+B], Pjd_3Ds[counter:counter+B] = fn_VisStd_OccStd(pd_amb_j2d, pd_amb_j3d, gt_Jvis)

        counter += B

    print(f'MPJPE: {MPJPEs.mean() :.3f}', end='  |  ')
    print(f'MPVPE: {MPVPEs.mean() :.3f}', end='  |  ')
    print(f'BH_MPJPE: {Bh_MPJPEs.mean() :.3f}', end='  |  ')
    print(f'BH_MPVPE: {Bh_MPVPEs.mean() :.3f}', end='  |  ' )
    print(f'AH: {Ahs.mean() :.3f}', end='  |  ' )
    print(f'PJD_2Dvis: {Pjd_2Ds.mean() :.3f}', end='  |  ' )
    print(f'PJD_3Docc: {Pjd_3Ds.mean() :.3f}', end='  |  ' )
    print(f'RD: {Pjd_2Ds.mean() / Pjd_3Ds.mean() :.3f}')



if __name__ == '__main__':

    snapshot_name = 'checkpoints/snapshots/HO3Dv3MHE_RD_0.33_fintuneWithEntropyLoss.pkl'
    snapshot = pickle.load(open(snapshot_name, 'rb'))['snapshot']
    num_samples = 201

    model = AmbHand(num_samples=num_samples)
    model.load_state_dict(snapshot)
    eval_HO3Dv3_MHE(model, stage='S2')

