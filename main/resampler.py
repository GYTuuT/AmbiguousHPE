
import os
import sys

sys.path.append(os.getcwd())


from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules.Attention import PositionalEmbedding, SelfAttentionBlock
from .modules.ResConv import ResidualConvBlock
from .modules.SemGcn import (ResSemanticGCN, get_AdjMat_from_edges,
                             get_hand_skeleton)


## ======================
class Frequency(nn.Module):
    def __init__(self, dim: int, n_levels: int = 8,
                **kwargs) -> None:
        """Positional encoding from NeRF: https://www.matthewtancik.com/nerf
        [sin(x), cos(x), sin(4x), cos(4x), sin(8x), cos(8x),
        ..., sin(2^n*x), cos(2^n*x)]
        """
        super().__init__()

        self.n_levels = n_levels
        assert self.n_levels > 0

        self.register_buffer('freqs', 2. ** torch.linspace(0., n_levels-1, n_levels),
                            persistent=False)

        self.in_dim = dim
        self.out_dim = dim * n_levels * 2


    # ---------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor): # [..., dim]
        x = x.unsqueeze(dim=-1) * self.freqs # (..., dim, n_levels)
        x = torch.cat((torch.sin(x), torch.cos(x)), dim=-1) # (..., dim, n_levels*2)
        return x.flatten(-2, -1) # (..., dim * n_levels * 2)




## ==========================
class JointWiseEnhanceLayer(nn.Module):
    def __init__(self,
                 image_features:List=[1024, 2048],
                 joint_features:int=128,
                 out_dim:int=64,
                 resolution:int=256,
                 **kwargs) -> None:
        super().__init__()

        self.num_joints = 21
        self.joint_features = joint_features
        self.resolution = resolution

        # 1. fusion and embed
        self.f4_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResidualConvBlock(image_features[-1], joint_features, channel_scale=2),
        )
        self.f3_layer = ResidualConvBlock(image_features[-2], joint_features, channel_scale=2)
        self.f3f4fuse_layer = ResidualConvBlock(joint_features*2, joint_features, channel_scale=2)

        self.f4_embed = nn.Sequential(
            nn.Linear(image_features[-1], joint_features),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(num_features=joint_features),
        )


        # 2. position encoding
        self.pos_encoding = Frequency(dim=3, n_levels=8) # output [..., 48]

        # 3. gcn
        num_gcn_blocks = 4
        _joint_bones = get_hand_skeleton(only_finger=False)
        _hand_adj = get_AdjMat_from_edges(self.num_joints, edges=_joint_bones,
                                          symmetric_norm=True)
        self.register_buffer('joint_bones', _joint_bones.to(dtype=torch.long))
        self.jointWiseGcn = ResSemanticGCN(self.f3f4fuse_layer.out_channels + self.pos_encoding.out_dim,
                                           hidden_features=256,
                                           out_features=joint_features,
                                           adj_matrix=_hand_adj, num_blocks=num_gcn_blocks,
                                           **kwargs)

        # 4. attention
        num_att_blocks = 6
        _pos_embed = PositionalEmbedding.SinCos1D(self.num_joints + 1, # joint feature + global feature
                                                  joint_features,
                                                  cls_token=False)
        self.register_buffer('pos_embed', _pos_embed)
        self.jointWiseAtt = nn.Sequential(*[SelfAttentionBlock(joint_features, **kwargs) for _ in range(num_att_blocks)],
                                          nn.LayerNorm(joint_features))


        # 5. out conv1d to compress feature
        self.out_layer = nn.Sequential(
            nn.Conv1d(joint_features, out_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_dim, out_dim, kernel_size=1, stride=1)
        )
        self.out_features = (self.num_joints + 1) * out_dim


    # -----------------------
    @staticmethod
    @torch.no_grad()
    def get_inner_sample_j2d(sampled_j2ds:Tensor,
                             heatmap_j2ds:Tensor,
                             heatmap_cfds:Tensor,
                             inner_threshold:float=10.0,
                             confid_threshold:float=0.25):
        """
        sampled_j2ds: [B, S(samples), 21, 2], reprojection of preds in image.
        heatmap_j2ds: [B, 21, 2], predict of heatmap.
        heatmap_cfds: [B, 21], confidence of heatmap.
        """

        dist_HmZi = torch.linalg.norm(heatmap_j2ds.unsqueeze(1) - sampled_j2ds, ord=2, dim=-1) # [B, S, 21]

        mask_zi = dist_HmZi < inner_threshold # high consistence
        mask_hm = heatmap_cfds > confid_threshold # high confidence (visiblity)
        inner_mask = torch.logical_and(mask_zi, mask_hm.unsqueeze(1)) # fuse of consistence and confidence, [B, S, 21]

        inner_ind = torch.argsort(inner_mask.sum(-1),
                                  dim=-1, descending=True)[:, 0] # [B], get the sample contains most inners, randomly select one if many samples meets it
        inner_j2d = sampled_j2ds.gather(dim=1, index=inner_ind.reshape(-1, 1, 1, 1).repeat(1, 1, 21, 2)).squeeze(1) # [B, 21, 2]
        inner_cfd = inner_mask.gather(dim=1, index=inner_ind.reshape(-1, 1, 1).repeat(1, 1, 21)).squeeze(1) # [B, 21]

        return inner_ind, inner_j2d, inner_cfd, inner_mask


    # ------------------------
    def forward(self,
                f3f4_features:List,
                sampled_j2ds:Tensor,
                heatmap_j2ds:Tensor,
                heatmap_cfds:Tensor):
        """
        f3f4_features: the last two feature maps out from backbone. [[B, 1024, 16, 16], [B, 2048, 8, 8]]
        sampled_j2ds: [B, S, 21, 2], j2d of generated samples.
        heatmap_j2ds: [B, 21, 2], j2d of heatmap prediction.
        heatmap_cfds: [B, 21], confidence of heatmap prediction.
        """

        B = sampled_j2ds.shape[0]


        # 1.sampling feature fuse
        fuse_feat = self.f3f4fuse_layer(
            torch.cat([
                self.f3_layer(f3f4_features[0]),
                self.f4_layer(f3f4_features[1])
            ], dim=1)
        ) # [B, J_features, 16, 16]
        global_feat = self.f4_embed(f3f4_features[1].mean(dim=(-1, -2))) # [B, 2048, 8, 8] -> [B, 2048] -> [B, J_features]

        # 3.joint feauture resmaple and concat
        _, inner_j2d, inner_cfd, _ = self.get_inner_sample_j2d(sampled_j2ds, heatmap_j2ds, heatmap_cfds) # [B, 21, 2], [B, 21]
        inner_uvs = ((inner_j2d / self.resolution) - 0.5) * 2.0
        bones_uvs = inner_j2d[:, self.joint_bones].mean(-2) # [B, 20, 2, 2] -> [B, 20, 2]

        J_feature = F.grid_sample(fuse_feat, inner_uvs.reshape(B, 1, -1, 2),
                                  mode='bilinear', align_corners=True).squeeze(-2) # [B, J_features, 21]
        B_feature = F.grid_sample(fuse_feat, bones_uvs.reshape(B, 1, -1, 2),
                                  mode='bilinear', align_corners=True).squeeze(-2) # [B, J_features, 20]

        feature = torch.cat([ # add bone and joint features
            (J_feature[..., :1] + B_feature[..., :5].mean(dim=-1, keepdim=True)) / 2.0, # root
            (J_feature[..., 1:] + B_feature) / 2.0,
        ], dim=-1).transpose(-1, -2).contiguous() # [B, J_features, 21] -> [B, 21, J_features]

        feature = torch.cat([ # cat the uv and confidence infomation with joint feature.
           feature,
           self.pos_encoding(torch.cat([inner_uvs, inner_cfd.unsqueeze(-1)], dim=-1)),
        ], dim=-1) # [B, 21, J_features + 48]

        # 4. gcn and attention
        feature = feature * inner_cfd[..., None] # mask the unreliable joint
        feature = self.jointWiseGcn(feature) # [B, 21, J_features]
        feature = torch.cat([feature, global_feat.unsqueeze(1)], dim=1) # [B, 22, J_features], cat the gloabl feature
        feature = self.jointWiseAtt(feature + self.pos_embed[None,...]) # [B, 22, J_features]

        # compress feature dims
        feature = self.out_layer(feature.transpose(-1, -2)).transpose(-1, -2)

        return feature.reshape(B, -1), (inner_j2d, inner_cfd)



