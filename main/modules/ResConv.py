

import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from torch import Tensor


##  ============
class ResidualConvBlock(nn.Module): # bottleneck residual

    def __init__(self, 
                 in_channels:int,
                 out_channels:int,
                 stride:int=1,
                 activation:nn.Module=nn.LeakyReLU,
                 normlayer:nn.Module=nn.BatchNorm2d,
                 channel_scale:int=4,
                 **kwargs) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_scale = channel_scale

        width = max(out_channels // self.channel_scale, 32)
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)
        self.norm1 = normlayer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1)
        self.norm2 = normlayer(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1)
        self.norm3 = normlayer(out_channels)

        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                normlayer(out_channels))
        
        self.activation = activation(inplace=True)
        
        self._init_weight()

    # --------
    def _init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv3.weight, mode="fan_out", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.downsample[0].weight, mode="fan_out", nonlinearity="leaky_relu")

        nn.init.constant_(self.norm1.weight, 1.), nn.init.constant_(self.norm1.bias, 0.)
        nn.init.constant_(self.norm2.weight, 1.), nn.init.constant_(self.norm2.bias, 0.)
        nn.init.constant_(self.norm3.weight, 1.), nn.init.constant_(self.norm3.bias, 0.)
        nn.init.constant_(self.downsample[1].weight, 1.)
        nn.init.constant_(self.downsample[1].bias, 0.)

    # ---------
    def forward(self, x:Tensor):

        assert x.shape[1] == self.in_channels, \
            RuntimeError(f'Expected input {x.shape} to have {self.in_channels} channels.')

        identity = x

        out = self.activation(self.norm1(self.conv1(x)))
        out = self.activation(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out = self.activation(out + identity)

        return out