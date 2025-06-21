
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn


## ========================
class SkipUpsampleBlock(nn.Module): # upsample decoder with skipped connecntion with encoder
    channel_scale = 2

    def __init__(self, 
                 feature_dims: List,  
                 out_channels: int,
                 activation: nn.Module=nn.LeakyReLU,
                 normlayer: nn.Module=nn.BatchNorm2d,
                 **kwargs) -> None:
        super().__init__()

        self.channel_list = feature_dims + [out_channels] # such as [2048, 1024, 512, 256, 21]
        self.num_layers = len(self.channel_list) - 1

        for idx, (cin, cout) in enumerate(zip(self.channel_list[:-1], self.channel_list[1:])):

            c_temp = max(cout // self.channel_scale, 64) # at least 64

            layer = nn.Sequential(
                nn.Conv2d(cin, c_temp, kernel_size=3, stride=1, padding=1, bias=False),
                normlayer(c_temp),
                activation(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(c_temp, cout, kernel_size=3, stride=1, padding=1, bias=False),
                normlayer(cout),
                activation(inplace=True),
            )
            setattr(self, f'layer_{idx}', layer)

        self.final_conv = nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False)


    # -------------
    def forward(self, features:List):

        assert len(features) == (self.num_layers), \
            ValueError(f'Wrong inputs feature dims, Input dims should be {self.channel_list[:-1]}')

        for i in range(self.num_layers):
            module = getattr(self, f'layer_{i}')

            if i == 0:
                x = module(features[i])
            else:
                x = module(x + features[i])

        x = self.final_conv(x)
        
        return x






## ============================
class SkipDeconvBlock(nn.Module):
    channel_scale = 2

    def __init__(self, 
                 feature_dims: List,  
                 out_channels: int,
                 activation: nn.Module=nn.LeakyReLU,
                 normlayer: nn.Module=nn.BatchNorm2d,
                 **kwargs) -> None:
        super().__init__()

        self.channel_list = feature_dims + [out_channels] # such as [2048, 1024, 512, 256, 21]
        self.num_layers = len(self.channel_list) - 1


        for idx, (cin, cout) in enumerate(zip(self.channel_list[:-1], self.channel_list[1:])):

            c_temp = max(cout // self.channel_scale, 64) # at least 64

            layer = nn.Sequential(
                nn.Conv2d(cin, c_temp, kernel_size=3, stride=1, padding=1, bias=False),
                normlayer(c_temp),
                activation(inplace=True),
                nn.ConvTranspose2d(c_temp, cout, kernel_size=4, stride=2, padding=1, bias=False),
                normlayer(cout),
                activation(inplace=True),
            )
            setattr(self, f'layer_{idx}', layer)

        self.final_conv = nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False)

    # -------------
    def forward(self, features:List):

        assert len(features) == (self.num_layers), \
            ValueError(f'Wrong inputs feature dims, Input dims should be {self.channel_list[:-1]}')

        for i in range(self.num_layers):
            module = getattr(self, f'layer_{i}')

            if i == 0:
                x = module(features[i])
            else:
                x = module(x + features[i])

        x = self.final_conv(x)

        return x




## =====================
class HeatmapPredictor(nn.Module):
    def __init__(self, 
                 num_peak:int, 
                 skip_feature_dims: List,
                 upsample_type:str='upsample', # 'upsample' or 'deconv'
                 **kwargs) -> None:
        super().__init__()

        assert upsample_type in ['upsample', 'deconv']

        upsampler = {'upsample': SkipUpsampleBlock,
                     'deconv':   SkipDeconvBlock,}[upsample_type]

        self.upsampler = upsampler(feature_dims=skip_feature_dims,
                                   out_channels=num_peak, 
                                   **kwargs)

    # ----------
    def forward(self, skip_features:List[Tensor]):
        return F.relu(self.upsampler(skip_features))


    # ----------
    @staticmethod
    @torch.no_grad()
    def construct_heatmap(reso:int, peaks:Tensor, sigma:float=1.0, hm_type:str='gaussian'):
        """
        reso: resolution of heatmap
        peaks: [B, J, 2], pixel location of peaks
        sigma: the std of gaussian.
        """
        assert hm_type in ['gaussian', 'laplacian'], ValueError('Invalid heatmap type.')

        B = peaks.shape[0]
        device =peaks.device
        heatmap = torch.stack(torch.meshgrid(torch.arange(reso, device=device),
                                             torch.arange(reso, device=device),
                                             indexing='xy'), dim=-1) # [H, W, 2]
        heatmap = heatmap[None,...] - peaks.view(-1, 1, 1, 2) # [N, H, W, 2]
        if hm_type == 'gaussian':
            heatmap = torch.exp(-heatmap.square().sum(-1) / (2 * sigma ** 2)) # [N, H, W]
        elif hm_type == 'laplacian':
            heatmap = torch.exp(-heatmap.abs().sum(-1) / sigma) # [N, H, W]

        return heatmap.reshape(B, -1, reso, reso)


    # --------
    @staticmethod
    @torch.no_grad()
    def get_peaks_uv(heatmap:Tensor) -> Tensor:
        """ Infer the uv postion of heatmap.
        Parameter
        ----
            heatmap: [B, C, H, W]
            threshold: float, threshold for decide the min confidece of heatmap.
        Return
        ----
            uv: [B, C, 2], xy coordinates in [0., 1.] plane, if invalid, value is -1.0.
        """
        B, C, H, W = heatmap.shape
        device = heatmap.device

        # smooth extreme points
        heatmap = F.avg_pool2d(heatmap, kernel_size=3, stride=1, padding=1)

        # get low confidence map
        confidence = heatmap.flatten(-2, -1).max(-1).values

        # get uvs        
        xy_prob = heatmap / (heatmap.sum(dim=(-1,-2), keepdim=True) + 1e-8) # [B, C, H, W]
        us = (xy_prob.sum(2) * torch.linspace(0.0, 1.0, steps=W, device=device).reshape(1, 1, -1)).sum(-1)
        vs = (xy_prob.sum(3) * torch.linspace(0.0, 1.0, steps=H, device=device).reshape(1, 1, -1)).sum(-1)

        return torch.stack([us, vs], dim=-1), confidence







if __name__ == '__main__':

    feats = [torch.randn([2, 256,  64, 64]),
             torch.randn([2, 512,  32, 32]),
             torch.randn([2, 1024, 16, 16]),
             torch.randn([2, 2048, 8,  8]),]

    m = HeatmapPredictor(num_peak=21, skip_feature_dims=[1024, 512, 256], upsample_type='upsample')

    n_parameters = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(n_parameters)

    y = m(feats[::-1][1:])

    exit()

    import cv2
    import numpy as np
    import torch

    points = torch.tensor([67.5, 67.5]).reshape(1, 1, 2)

    g_heatmap = HeatmapPredictor.construct_heatmap(128, points, sigma=7.5, hm_type='gaussian')
    l_heatmap = HeatmapPredictor.construct_heatmap(128, points, sigma=7.5, hm_type='laplacian')

    g_heatmap = (g_heatmap.squeeze().numpy() * 255).astype(np.uint8) 
    g_heatmap = cv2.applyColorMap(g_heatmap, colormap=cv2.COLORMAP_MAGMA) # [H, W] -> [H, W, 3]
    l_heatmap = (l_heatmap.squeeze().numpy() * 255).astype(np.uint8) 
    l_heatmap = cv2.applyColorMap(l_heatmap, colormap=cv2.COLORMAP_MAGMA) # [H, W] -> [H, W, 3]

    gl_compare_map = np.concatenate([g_heatmap, l_heatmap], axis=1)
    cv2.imwrite('g_l_heatmap_compare.png', gl_compare_map)
