
from typing import Dict, List

import torch
from torch import Tensor

from tools.trainingUtils import LossBase
from tools.rotationConvert import batch_rotMat2R6d_tensor
from .modules.Upsample import HeatmapPredictor


## ===================
class AmbHandLoss(LossBase):
    def __init__(self,
                 basename:str=None) -> None:
        super().__init__(basename=basename)


    # --------
    @LossBase.register_loss('nll')
    def fn_nll(self, log_prob:Tensor, hand_valid:Tensor=None):
        """
        log_prob: [B, N]
        hand_valid: [B,], 0/1 mask, 1 for valid, 0 for invalid
        """
        assert log_prob.ndim == 2

        if hand_valid is not None:
            valid = hand_valid.reshape(-1, 1).expand(-1, log_prob.shape[1])
            err = - (log_prob * valid).sum() / (valid.sum() + 1e-8)
        else:
            err = - log_prob.mean()

        return err # per sample

    # --------
    @LossBase.register_loss('reg')
    def fn_r6d_reg(self, pred_r6d:Tensor):
        """
        pred_r6d: [B, N, 6]
        """
        assert pred_r6d.ndim == 3 or pred_r6d.ndim == 4

        r6d = pred_r6d.reshape(-1, 3, 2)
        r6d_mm = torch.matmul(r6d.transpose(-1, -2), r6d)
        I = torch.eye(2, device=r6d.device, dtype=r6d.dtype)[None,...] # [1, 2, 2]

        return (r6d_mm - I).square().sum((-2, -1)).mean() # per J

    # --------
    @LossBase.register_loss('z0r6d_l2')
    def fn_z0_r6d_err(self, pred_z0r6d:Tensor, gt_r6d:Tensor, hand_valid:Tensor=None):
        """
        pred_z0_r6d: [B, J, 6]
        gt_r6d: [B, J, 6]
        hand_valid: [B,], 0/1 mask, 1 for valid, 0 for invalid
        """
        assert pred_z0r6d.ndim == 3
        assert gt_r6d.ndim == 3

        err = (pred_z0r6d - gt_r6d).square().sum(-1) # [B, J]
        if hand_valid is not None:
            valid = hand_valid.reshape(-1, 1).expand(-1, err.shape[1])
            err = (err * valid).sum() / (valid.sum() + 1e-8)
        else:
            err = err.mean()

        return err

    # ---------
    @LossBase.register_loss('z0j3d')
    def fn_z0_j3d_err(self, pred_z0j3d:Tensor, gt_j3d:Tensor, joint_valid:Tensor=None):
        """
        pred_z0j3d: [B, 21, 3]
        gt_j3d: [B, 21, 3]
        joint_valid: [B, 21], 0/1 mask, 1 for valid, 0 for invalid
        """
        assert pred_z0j3d.ndim == 3
        assert gt_j3d.ndim == 3

        ra_pred_z0j3d = (pred_z0j3d - pred_z0j3d[:, :1])  # root align
        ra_gt_j3d = (gt_j3d - gt_j3d[:, :1])
        err = ((ra_pred_z0j3d - ra_gt_j3d).square().sum(-1) + 1e-8).sqrt() # [B, 21]

        if joint_valid is not None:
            valid = joint_valid.reshape(-1, err.shape[1])
            err = (err * valid).sum() / (valid.sum() + 1e-8)
        else:
            err = err.mean()

        return err # per J

    # ---------
    @LossBase.register_loss('z0v3d')
    def fn_z0_v3d_err(self, ra_pred_z0v3d:Tensor, ra_gt_v3d:Tensor, hand_valid:Tensor=None):
        """
        ra_pred_z0v3d: [B, 778, 3], root_aligned verts3d
        ra_gt_v3d: [B, 778, 3], root_aligned verts3d
        hand_valid: [B,], 0/1 mask, 1 for valid, 0 for invalid
        """
        assert ra_pred_z0v3d.ndim == 3
        assert ra_gt_v3d.ndim == 3

        err = ((ra_pred_z0v3d - ra_gt_v3d).square().sum(-1) + 1e-8).sqrt() # [B, 778]
        if hand_valid is not None:
            valid = hand_valid.reshape(-1, 1).expand(-1, err.shape[1])
            err = (err * valid).sum() / (valid.sum() + 1e-8)
        else:
            err = err.mean()

        return err # per V

    # ---------
    @LossBase.register_loss('minj3d')
    def fn_min_j3d_err(self, pred_j3d:Tensor, gt_j3d:Tensor, hand_valid:Tensor=None):
        """
        pred_j3d: [B, N, 21, 3]
        gt_j3d: [B, 21, 3]
        hand_valid: [B,], 0/1 mask, 1 for valid, 0 for invalid
        """
        assert pred_j3d.ndim == 4
        assert gt_j3d.ndim == 3

        ra_pred_j3d, ra_gt_j3d = (pred_j3d - pred_j3d[..., :1, :]), (gt_j3d - gt_j3d[..., :1, :]) # root align
        dist = ((ra_pred_j3d - ra_gt_j3d.unsqueeze(1)).square().sum(-1) + 1e-8).sqrt().mean(-1) # [B, N]
        dist, _ = torch.sort(dist, dim=1) # sort in small to large

        if hand_valid is not None:
            valid = hand_valid.reshape(-1)
            err = (dist[:, 0] * valid).sum() / (valid.sum() + 1e-8)
        else:
            err = dist[:, 0].mean()

        return err

    # ---------
    @LossBase.register_loss('visj2d')
    def fn_vis_j2d_err(self, pred_j2d:Tensor, gt_j2d:Tensor, gt_Jvis:Tensor, hand_valid:Tensor=None):
        """
        pred_j2d: [B, N, 21, 2]
        gt_j2d: [B, 21, 2]
        gt_Jvis: [B, 21]
        hand_valid: [B,], 0/1 mask, 1 for valid, 0 for invalid
        """
        assert pred_j2d.ndim == 4
        assert gt_j2d.ndim == 3
        assert gt_Jvis.ndim == 2

        diff2d = pred_j2d - gt_j2d.unsqueeze(1) # [B, N, 21, 2]
        diff2d = (diff2d.square().sum(-1) + 1e-8).sqrt().mean(1) # [B, 21]
        vis_diff = (diff2d * gt_Jvis).sum(-1) / (gt_Jvis.sum(-1) + 1e-8) # [B], reduce joint dimension.

        if hand_valid is not None:
            valid = hand_valid.reshape(-1)
            err = (vis_diff * valid).sum() / (valid.sum() + 1e-8)
        else:
            err = vis_diff.mean()

        return err

    # # ---------
    # @LossBase.register_loss('visstd')
    # def fn_vis_std_err(self, pred_j3d:Tensor, gt_Jvis:Tensor, hand_valid:Tensor=None):
    #     """
    #     pred_j3d: [B, N, 21, 3]
    #     gt_Jvis: [B, 21]
    #     tgt_std: float, target std of occluded joint3d, comes from the statistics of real data.
    #     hand_valid: [B,], 0/1 mask, 1 for valid, 0 for invalid
    #     """
    #     assert pred_j3d.ndim == 4
    #     assert gt_Jvis.ndim == 2

    #     diff3d = pred_j3d - pred_j3d.mean(dim=1, keepdim=True).detach() # [B, N, 21, 3]
    #     diff3d = (diff3d.square().sum(-1) + 1e-8).sqrt().mean(1) # [B, 21]
    #     vis_std = (diff3d * gt_Jvis).sum(-1) / (gt_Jvis.sum(-1) + 1e-8) # [B], reduce joint dimension.

    #     if hand_valid is not None:
    #         valid = hand_valid.reshape(-1)
    #         vis_std = (vis_std * valid).sum() / (valid.sum() + 1e-8)
    #     else:
    #         vis_std = vis_std.mean()

    #     return vis_std.abs()

    # ---------
    @LossBase.register_loss('occstd')
    def fn_occ_std_err(self, pred_j3d:Tensor, gt_Jvis:Tensor, tgt_std:float=20.0, hand_valid:Tensor=None):
        """
        pred_j3d: [B, N, 21, 3]
        gt_Jvis: [B, 21]
        tgt_std: float, target std of occluded joint3d, comes from the statistics of real data.
        hand_valid: [B,], 0/1 mask, 1 for valid, 0 for invalid
        """
        assert pred_j3d.ndim == 4
        assert gt_Jvis.ndim == 2

        gt_Jocc = gt_Jvis.logical_not()
        diff3d = pred_j3d - pred_j3d.mean(dim=1, keepdim=True).detach() # [B, N, 21, 3]
        diff3d = (diff3d.square().sum(-1) + 1e-8).sqrt().mean(1) # [B, 21]
        occ_std = (diff3d * gt_Jocc).sum(-1) / (gt_Jocc.sum(-1) + 1e-8) # [B], reduce joint dimension.

        if hand_valid is not None:
            valid = hand_valid.reshape(-1)
            occ_std = (occ_std * valid).sum() / (valid.sum() + 1e-8)
        else:
            occ_std = occ_std.mean()

        return (tgt_std - occ_std).abs()

    # ---------
    @LossBase.register_loss('entropy') # the implement of max entropy loss in MHEntropy, ICCV'23
    def fn_entropy(self, sampled_log_p:Tensor):
        """
        pred_log_p: [B, N]
        """
        return sampled_log_p.abs().mean() # per sample, see: https://github.com/GloryyrolG/MHEntropy/tree/master

    # ---------
    @LossBase.register_loss('hm_l2')
    def fn_heatmap_err(self, pred_heatmap:Tensor, gt_heatmap:Tensor, joint_valid:Tensor=None):
        """
        pred_heatmap: [B, J, H, W]
        gt_heatmap: [B, J, H, W]
        joint_valid: [B, J], 0/1 mask, 1 for valid, 0 for invalid
        """
        assert pred_heatmap.ndim == 4
        assert gt_heatmap.ndim == 4

        err = (pred_heatmap - gt_heatmap).square().sum((-1,-2)) # [B, J]
        if joint_valid is not None:
            valid = joint_valid
            err = (err * valid).sum() / (valid.sum() + 1e-8)
        else:
            err = err.mean()

        return err # per J




    # -----------
    def compute_loss_stage_1(self, preds:Dict, gts:Dict,
                             loss_weights:List=[0.0]*3,):

        if preds is None:
            return 0

        self.basename = 'S1'

        hand_valid  = gts.get('hand_valid', None)
        joint_valid = gts.get('joint_valid', None)

        loss_nll = self.fn_nll(preds['log_p'], hand_valid)
        loss_r6dreg = self.fn_r6d_reg(preds['amb_r6d'])



        gt_Jvis = gts['joint_vis'].reshape(-1, 20)
        gt_Jvis = torch.cat([torch.ones_like(gt_Jvis[:, :1]), gt_Jvis], dim=-1)

        self.fn_z0_j3d_err(preds['z0_j3d'] * 1000, gts['joint3d'] * 1000, joint_valid) # -> mm
        self.fn_vis_j2d_err(preds['amb_j2d'], gts['joint2d'], gt_Jvis, hand_valid)
        self.fn_occ_std_err(preds['amb_j3d'] * 1000, gt_Jvis, 10.0, hand_valid)

        loss_heatmap = 0.0
        if preds.get('heatmap', None) is not None:

            reso_pred = preds['heatmap'].shape[-1]
            reso_gt   = gts['image'].shape[-1]
            scale, sigma = reso_pred / reso_gt, reso_pred * 0.05

            gt_heatmap = HeatmapPredictor.construct_heatmap( # only finger joints
                reso_pred, peaks=gts['joint2d'] * scale, sigma=sigma, hm_type='laplacian') # [B, 21, H, W]
            root_vis = gts['joint_vis'].reshape(-1, 20).any(dim=-1, keepdim=True)
            all_vis = torch.cat([root_vis, gts['joint_vis'].reshape(-1, 20)], dim=-1).reshape(-1, 21, 1, 1)

            loss_heatmap = self.fn_heatmap_err(preds['heatmap'], (gt_heatmap * all_vis).detach()) # mask the unseen joints


        loss_nll     = loss_weights[0] * loss_nll
        loss_r6dreg  = loss_weights[1] * loss_r6dreg
        loss_heatmap = loss_weights[2] * loss_heatmap

        loss = loss_nll + loss_r6dreg + loss_heatmap

        return loss




    # # --------------
    # def compute_loss_stage_2(self, preds:Dict, gts:Dict,
    #                         loss_weights:List=[0.0]*6,):

    #     if preds is None:
    #         return 0

    #     self.basename = 'S2'

    #     hand_valid = gts.get('hand_valid', None)
    #     joint_valid = gts.get('joint_valid', None)

    #     loss_nll = self.fn_nll(preds['log_p'], hand_valid)
    #     loss_r6dreg = self.fn_r6d_reg(preds['amb_r6d'])

    #     gt_rmt = torch.cat([gts['root_rmt'], gts['hand_rmt']], dim=1) # [B, 16, 3, 3]
    #     gt_r6d = batch_rotMat2R6d_tensor(gt_rmt.reshape(-1, 3, 3)).reshape(-1, 16, 6)
    #     loss_r6d = self.fn_z0_r6d_err(preds['z0_r6d'], gt_r6d, hand_valid)

    #     loss_z0j3d = self.fn_z0_j3d_err(preds['z0_j3d'] * 1000, gts['joint3d'] * 1000, joint_valid) # -> mm

    #     gt_Jvis = gts['joint_vis'].reshape(-1, 20)
    #     gt_Jvis = torch.cat([torch.ones_like(gt_Jvis[:, :1]), gt_Jvis], dim=-1)
    #     loss_visj2d = self.fn_vis_j2d_err(preds['amb_j2d'], gts['joint2d'], gt_Jvis, hand_valid)
    #     loss_occstd = self.fn_occ_std_err(preds['amb_j3d'] * 1000, gt_Jvis, 16.0, hand_valid)

    #     loss_nll    = loss_weights[0] * loss_nll
    #     loss_r6dreg = loss_weights[1] * loss_r6dreg

    #     loss_r6d    = loss_weights[2] * loss_r6d
    #     loss_z0j3d  = loss_weights[3] * loss_z0j3d

    #     loss_visj2d = loss_weights[4] * loss_visj2d
    #     loss_occstd = loss_weights[5] * loss_occstd

    #     loss = loss_nll + loss_r6dreg + loss_r6d + loss_z0j3d + loss_visj2d + loss_occstd

    #     return loss



    # --------------
    def compute_loss_stage_2(self, preds:Dict, gts:Dict,
                            loss_weights:List=[0.0]*6,):

        if preds is None:
            return 0

        self.basename = 'S2'

        hand_valid = gts.get('hand_valid', None)
        joint_valid = gts.get('joint_valid', None)

        loss_nll = self.fn_nll(preds['log_p'], hand_valid)
        loss_r6dreg = self.fn_r6d_reg(preds['amb_r6d'])

        gt_rmt = torch.cat([gts['root_rmt'], gts['hand_rmt']], dim=1) # [B, 16, 3, 3]
        gt_r6d = batch_rotMat2R6d_tensor(gt_rmt.reshape(-1, 3, 3)).reshape(-1, 16, 6)
        loss_r6d = self.fn_z0_r6d_err(preds['z0_r6d'], gt_r6d, hand_valid)

        loss_z0j3d = self.fn_z0_j3d_err(preds['z0_j3d'] * 1000, gts['joint3d'] * 1000, joint_valid) # -> mm

        gt_Jvis = gts['joint_vis'].reshape(-1, 20)
        gt_Jvis = torch.cat([torch.ones_like(gt_Jvis[:, :1]), gt_Jvis], dim=-1)
        loss_visj2d = self.fn_vis_j2d_err(preds['amb_j2d'], gts['joint2d'], gt_Jvis, hand_valid)
        loss_entropy = self.fn_entropy(preds['log_p_sampled']) # [B, N]

        loss_nll    = loss_weights[0] * loss_nll
        loss_r6dreg = loss_weights[1] * loss_r6dreg
        loss_r6d    = loss_weights[2] * loss_r6d
        loss_z0j3d  = loss_weights[3] * loss_z0j3d

        loss_visj2d  = loss_weights[4] * loss_visj2d
        loss_entropy = loss_weights[5] * loss_entropy

        loss = loss_nll + loss_r6dreg + loss_r6d + loss_z0j3d + loss_visj2d + loss_entropy

        return loss