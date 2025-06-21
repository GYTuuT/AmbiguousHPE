from typing import Any, Dict, List, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tools.dataProcessing import get_procrustes_alignment_tensor

from .backbone.Resnet_50 import resnet50
from .flowPredictor import GlowRmtBetaCamPredictor, GlowRmtPredictor
from .modules.Manolayer import ManoBasic, RotMatrixManolayer
from .modules.Upsample import HeatmapPredictor
from .resampler import JointWiseEnhanceLayer

# ----------------
NUM_MANO_J = 16
NUM_SKELETON_J = 21
WEAK_CAMERA_BASE = {'avg':[1.5, 120.0, 120.0], 'std':[0.25, 30.0, 30.0],} # obtained from the training data.
MANO_MODELS = 'checkpoints/processed_mano'
FEATURE_CHANNELS = [256, 512, 1024, 2048] # channels of feature maps that out from backbone


# ==================
class AmbHandStageBase(nn.Module):
    def __init__(self,
                 num_samples:int,
                 manolayer: RotMatrixManolayer,
                 **kwargs) -> None:
        super().__init__()

        self.num_samples = num_samples
        self.rManolayer = manolayer

    # ------------
    def forward(self, image_features:List, batch_data:Dict) -> Dict:
        """Model Forward Pass.

        Parameters
        ----
        image_features: feature maps list out from backnone.

        batch_data:
            inputs containing the following necessary keyword:

            images : Tensor
                `[B, 3, H, W]`
            root_rmt : Tensor
                `[B, 1, 3, 3]`, gt root rmt
            amb_finger_rmt : Tensor
                `[B, N_gt, 15, 3, 3]`, gt multi-hypothesis finger rmt samples

        Returns
        ----
        preds : Dict
            outputs following below format.
        """
        raise NotImplementedError('Not Implemented')


## ===================
class AmbHand_Stage_1(AmbHandStageBase):
    def __init__(self,
                 num_samples:int,
                 manolayer: RotMatrixManolayer,
                 use_amb_anno:bool=True,
                 **kwargs) -> None:
        super().__init__(num_samples, manolayer, **kwargs)

        # stage 1 predict: heatmap, hand pose rmt, hand shape, weak-cam.

        self.use_amb_anno = use_amb_anno

        self.heatmapPredictor = HeatmapPredictor(NUM_SKELETON_J,
                                                 skip_feature_dims=FEATURE_CHANNELS[::-1][1:])

        self.flowPredictor = GlowRmtBetaCamPredictor(context_features=FEATURE_CHANNELS[-1],
                                                     num_joints=NUM_MANO_J,
                                                     hidden_features=512,
                                                     num_blocks_per_layer=2,
                                                     num_layers=4,
                                                     camera_base=WEAK_CAMERA_BASE,
                                                     )

    # --------
    def forward(self, image_features: List[Tensor], batch_data: Dict) -> Dict:

        context = image_features[-1].mean(dim=(-1, -2))
        B, N = context.shape[0], self.num_samples
        img_resolution = batch_data['image'].shape[-1]

        # 1. get heatmap
        heatmaps = self.heatmapPredictor(image_features[::-1][1:])
        hm_uvs, hm_cfd = self.heatmapPredictor.get_peaks_uv(heatmaps) # [B, 21, 2], [B, 21]

        # 2. get flow results
        pred_rmt, pred_r6d, pred_shape, pred_weakcam, sampled_logp \
            = self.flowPredictor.generate_random_samples(ctx=context, num_samples=N)

        if self.training:
            N_gt = batch_data['multi_rmt'].shape[1]
            if self.use_amb_anno:
                gt_amb_finger_rmt = batch_data['multi_rmt'].reshape(B, -1, 15, 3, 3)
            else:
                gt_amb_finger_rmt = batch_data['hand_rmt'].reshape(B, 1, 15, 3, 3).repeat(1, N_gt, 1, 1, 1)
            gt_root_rmt = batch_data['root_rmt'].reshape(B, 1, 1, 3, 3).repeat(1, N_gt, 1, 1, 1)
            gt_amb_rmt = torch.cat([gt_root_rmt, gt_amb_finger_rmt], dim=2) # [B, N_gt, 16, 3, 3]

            gt_shape = batch_data['hand_shape']
            gt_weakcam = batch_data['weakcam']

            pred_logp, _ = self.flowPredictor.get_inputs_log_prob(
                ctx=context,
                rmt=gt_amb_rmt,
                beta=gt_shape.unsqueeze(1).repeat(1, N_gt, 1),
                cam=gt_weakcam.unsqueeze(1).repeat(1, N_gt, 1),
            )
        else:
            pred_logp = torch.zeros(B, N).to(pred_rmt.device)

        # 4. get predicted joints
        _amb_rmt = pred_rmt.reshape(-1, 16, 3, 3) # [B*N, 16, 3, 3]
        _amb_cam = pred_weakcam.reshape(-1, 1, 3) # [B*N, 1, 3]
        _amb_shape = pred_shape.reshape(-1, 10) # [B*N, 10]
        pred_amb_v3d, pred_amb_j3d = self.rManolayer(_amb_rmt[:, :1], _amb_rmt[:, 1:], _amb_shape) # [B*N, 21, 3]
        pred_amb_j2d = (pred_amb_j3d[...,:2] * 1000) * _amb_cam[...,:1] + _amb_cam[...,1:] # j x s + t


        # 5. arange outputs
        pred_logp = pred_logp.reshape(B, -1)
        pred_shape = pred_shape.reshape(B, N, 10)
        pred_weakcam = pred_weakcam.reshape(B, N, 3)
        pred_rmt = pred_rmt.reshape(B, N, 16, 3, 3)
        pred_r6d = pred_r6d.reshape(B, N, 16, 6)
        pred_amb_v3d = pred_amb_v3d.reshape(B, N, 778, 3)
        pred_amb_j3d = pred_amb_j3d.reshape(B, N, 21, 3)
        pred_amb_j2d = pred_amb_j2d.reshape(B, N, 21, 2)
        hm_j2d = (hm_uvs * img_resolution).reshape(B, 21, 2)
        hm_cfd = hm_cfd.reshape(B, 21)

        preds = dict(
            log_p = pred_logp,
            log_p_sampled = sampled_logp,

            # multi hypo
            amb_shape = pred_shape,# [B, N_pred, 10]
            amb_weakcam = pred_weakcam, # [B, N_pred, 3]
            amb_rmt = pred_rmt, # [B, N_pred, 16, 3, 3]
            amb_r6d = pred_r6d, # [B, N_pred, 16, 6]
            amb_v3d = pred_amb_v3d, # [B, N_pred, 778, 3]
            amb_j3d = pred_amb_j3d, # [B, N_pred, 21, 3], in meters
            amb_j2d = pred_amb_j2d, # [B, N_pred, 21, 2], in pixel

            # z0 pred
            z0_shape = pred_shape[:, 0], # [B, 10]
            z0_weakcam = pred_weakcam[:, 0], # [B, 3]

            z0_rmt = pred_rmt[:, 0], # [B, 16, 3, 3], the first sample of rmtFlow is always z0 sample
            z0_r6d = pred_r6d[:, 0], # [B, 16, 6]
            z0_v3d = pred_amb_v3d[:, 0], # [B, 778, 3]
            z0_j3d = pred_amb_j3d[:, 0], # [B, 21, 3], in meters
            z0_j2d = pred_amb_j2d[:, 0], # [B, 21, 2], in pixel

            # heatmap pred
            heatmap = heatmaps,
            hm_j2d = hm_j2d, # [B, 21, 2]
            hm_cfd = hm_cfd, # [B, 21]
        )

        return preds





## ===================
class AmbHand_Stage_2(AmbHandStageBase):
    def __init__(self,
                 num_samples: int,
                 manolayer: RotMatrixManolayer,
                 use_amb_anno:bool=True,
                 **kwargs) -> None:
        super().__init__(num_samples, manolayer, **kwargs)

        self.use_amb_anno = use_amb_anno

        self.jointWiseResampler = JointWiseEnhanceLayer(image_features=FEATURE_CHANNELS[-2:],
                                                        joint_features=256,
                                                        out_dim=64)

        self.flowPredictor = GlowRmtPredictor(context_features=self.jointWiseResampler.out_features,
                                              num_joints=NUM_MANO_J,
                                              hidden_features=512,
                                              num_blocks_per_layer=2,
                                              num_layers=4,
                                              noise_scale=1e-3)


    # --------
    def forward(self, image_features: List, batch_data: Dict, stage1_preds:Dict) -> Dict:

        # NOTE IMPORTANT no gradient which backprop from Stage 2 for Stage 1 except the backbone.
        f3f4_features = image_features[2:]
        sampled_j2ds = stage1_preds['amb_j2d'].detach() # [B, N_pred, 21, 2], cut the gradient with stage1's output.
        heatmap_j2ds = stage1_preds['hm_j2d'].detach() # [B, 21, 2]
        heatmap_cfds = stage1_preds['hm_cfd'].detach() # [B, 21]
        weakcam = stage1_preds['z0_weakcam'].detach() # [B, 3], all use the z0 sample.
        shape = stage1_preds['z0_shape'].detach() # [B, 10]

        B, N = sampled_j2ds.shape[0], self.num_samples
        context, (inner_j2d, inner_cfd) = self.jointWiseResampler(f3f4_features,
                                                                  sampled_j2ds,
                                                                  heatmap_j2ds,
                                                                  heatmap_cfds)

        # 1. get hand pose rot-matrix
        pred_rmt, pred_r6d, sampled_logp = \
            self.flowPredictor.generate_random_samples(ctx=context, num_samples=N) # [B, N, ...], the first is z0 sampel.

        if self.training:
            N_gt = batch_data['multi_rmt'].shape[1]
            if self.use_amb_anno:
                gt_amb_finger_rmt = batch_data['multi_rmt'].reshape(B, -1, 15, 3, 3)
            else:
                gt_amb_finger_rmt = batch_data['hand_rmt'].reshape(B, 1, 15, 3, 3).repeat(1, N_gt, 1, 1, 1)
            gt_root_rmt = batch_data['root_rmt'].reshape(B, 1, 1, 3, 3).repeat(1, N_gt, 1, 1, 1)
            gt_amb_rmt = torch.cat([gt_root_rmt, gt_amb_finger_rmt], dim=2) # [B, N_gt, 16, 3, 3]

            pred_logp, _ = self.flowPredictor.get_inputs_log_prob(ctx=context, rmt=gt_amb_rmt)
        else:
            pred_logp = torch.zeros(B, N).to(pred_rmt.device)

        # 2. get mano joints, use stage1's weak-cam and shape.
        _amb_rmt = pred_rmt.reshape(-1, 16, 3, 3) # [B*N, 16, 3, 3]
        _amb_cam = weakcam.unsqueeze(1).repeat(1, N, 1).reshape(-1, 1, 3) # [B*N, 1, 3]
        _amb_shape = shape.unsqueeze(1).repeat(1, N, 1).reshape(-1, 10) # [B*N, 10]
        pred_amb_v3d, pred_amb_j3d = self.rManolayer(_amb_rmt[:, :1], _amb_rmt[:, 1:], _amb_shape) # [B*N, 21, 3]
        pred_amb_j2d = (pred_amb_j3d[...,:2] * 1000) * _amb_cam[...,:1] + _amb_cam[...,1:] # j x s + t

        # 5. arange
        pred_logp = pred_logp.reshape(B, -1)
        pred_rmt = pred_rmt.reshape(B, N, 16, 3, 3)
        pred_r6d = pred_r6d.reshape(B, N, 16, 6)
        pred_amb_v3d = pred_amb_v3d.reshape(B, N, 778, 3)
        pred_amb_j3d = pred_amb_j3d.reshape(B, N, 21, 3)
        pred_amb_j2d = pred_amb_j2d.reshape(B, N, 21, 2)
        _amb_shape = _amb_shape.reshape(B, N, 10)
        _amb_cam   = _amb_cam.reshape(B, N, 3)

        preds = dict(
            log_p = pred_logp,
            log_p_sampled = sampled_logp,

            # multi hypo
            amb_shape = _amb_shape,# [B, N_pred, 10]
            amb_weakcam = _amb_cam, # [B, N_pred, 3]

            amb_rmt = pred_rmt, # [B, N_pred, 16, 3, 3]
            amb_r6d = pred_r6d, # [B, N_pred, 16, 6]
            amb_v3d = pred_amb_v3d, # [B, N_pred, 778, 3]
            amb_j3d = pred_amb_j3d, # [B, N_pred, 21, 3], in meters
            amb_j2d = pred_amb_j2d, # [B, N_pred, 21, 2], in pixel

            # z0 pred
            z0_shape = _amb_shape[:, 0], # [B, 10]
            z0_weakcam = _amb_cam[:, 0], # [B, 3]

            z0_rmt = pred_rmt[:, 0], # [B, 16, 3, 3], the first sample of rmtFlow is always z0 sample
            z0_r6d = pred_r6d[:, 0], # [B, 16, 6]
            z0_v3d = pred_amb_v3d[:, 0], # [B, 778, 3]
            z0_j3d = pred_amb_j3d[:, 0], # [B, 21, 3], in meters
            z0_j2d = pred_amb_j2d[:, 0], # [B, 21, 2], in pixel

            # inners
            inner_j2d = inner_j2d, # [B, 21, 2]
            inner_cfd = inner_cfd, # [B, 21]
        )

        return preds


## =================
class AmbHand(nn.Module):
    def __init__(self,
                 num_samples:int,
                 frozen_backbone:bool=False,
                 frozen_stage_1:bool=False,
                 frozen_stage_2:bool=False,
                 s1_use_amb_anno:bool=False, # fixme: if using multi-hypotheses annos in S1
                 s2_use_amb_anno:bool=True, # fixme: if using multi-hypotheses annos in S2
                 **kwargs) -> None:
        super().__init__()

        self.name = 'AmbHandNet'
        self.frozen_backbone = frozen_backbone
        self.frozen_stage_1 = frozen_stage_1
        self.frozen_stage_2 = frozen_stage_2

        right_manolayer = RotMatrixManolayer(MANO_MODELS, hand_type='right')

        self.backbone = resnet50(pretrained=True)
        self.stage_1_modules = AmbHand_Stage_1(num_samples, right_manolayer, s1_use_amb_anno, **kwargs)
        self.stage_2_modules = AmbHand_Stage_2(num_samples, right_manolayer, s2_use_amb_anno, **kwargs)

        self.frozen_layer_params(self.backbone) if frozen_backbone else ...
        self.frozen_layer_params(self.stage_1_modules) if frozen_stage_1 else ...
        self.frozen_layer_params(self.stage_2_modules) if frozen_stage_2 else ...


    # -----------
    def forward(self, batch_data:Dict, only_stage_1:bool=False):

        image_features = self.backbone.forward_features(batch_data['image'])

        stage_1_preds = self.stage_1_modules(image_features, batch_data)

        if only_stage_1:
            stage_2_preds = None
        else:
            stage_2_preds = self.stage_2_modules(image_features, batch_data, stage_1_preds)

        return stage_1_preds, stage_2_preds


    # ------------
    def frozen_layer_params(self, layer:nn.Module):
        layer.requires_grad_(False)
        for name, param in layer.named_parameters():
            param.requires_grad = False
        for name, buffer in layer.named_buffers():
            buffer.requires_grad = False

    # ------------
    def add_v3d_weakcam_for_gts(self, batch_data:Dict):

        src_v3d, src_j3d = self.stage_1_modules.rManolayer(batch_data['root_rmt'],
                                                           batch_data['hand_rmt'],
                                                           batch_data['hand_shape']) # [B, 778, 3], [B, 21, 3]
        align_j2d, rst = get_procrustes_alignment_tensor(src_j3d[..., :2] * 1000,
                                                         batch_data['joint2d'], exclude_R=True) # PA align

        weakcam = torch.cat([rst[1].reshape(-1, 1), rst[2].reshape(-1, 2)], dim=-1) # [B, 3]
        align_v3d = src_v3d - src_j3d[..., :1, :]
        batch_data.update(dict(ra_verts3d=align_v3d.detach(), weakcam=weakcam.detach()))

        return batch_data
