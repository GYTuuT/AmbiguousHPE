
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nflows.customs.pmade import PatchMADE
from nflows.transforms.base import Transform
from nflows.utils import torchutils
import math



## ===========================
class PatchAutoregressiveTransform(Transform):

    def __init__(self,
                 autoregressive_net:nn.Module,
                 patch_size:int,
                 **kwargs) -> None:
        super(PatchAutoregressiveTransform, self).__init__()

        self.autoregressive_net = autoregressive_net
        self.patch_size = patch_size

    def forward(self, inputs, context=None):
        autoregressive_params = self.autoregressive_net(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        num_inputs = np.prod(inputs.shape[1:])
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        loops = num_inputs // self.patch_size

        for _ in range(loops):
            autoregressive_params = self.autoregressive_net(outputs, context)
            outputs, logabsdet = self._elementwise_inverse(
                inputs, autoregressive_params
            )

        return outputs, logabsdet



## =========================
class PatchMaskedAffineAutoregressiveTransform(PatchAutoregressiveTransform):
    def __init__(
            self,
            features: int,
            hidden_features: int,
            context_features: int=None,
            patch_size: int=1,
            num_blocks: int=2,

            dropout_rate: float=0.,
            activation: nn.Module=nn.LeakyReLU,
            normlayer: nn.Module=None,
            ):

        self.features = features
        self._eps = 1e-3

        pmade = PatchMADE(
            in_features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            patch_size=patch_size,
            out_multiplier=self._output_dim_multiplier(),
            num_blocks=num_blocks,

            dropout_rate=dropout_rate,
            activation=activation,
            normlayer=normlayer
        )

        super(PatchMaskedAffineAutoregressiveTransform, self).__init__(
            autoregressive_net=pmade, patch_size=patch_size)


    def _output_dim_multiplier(self):
        return 2


    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        scale = F.softplus(unconstrained_scale) + self._eps
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet


    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        scale = F.softplus(unconstrained_scale) + self._eps
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet


    def _unconstrained_scale_and_shift(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier())
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        return unconstrained_scale, shift






def main():

    inputs = torch.randn([2, 8])
    context = torch.randn([2, 2])
    transform = PatchMaskedAffineAutoregressiveTransform(
        features=8,
        hidden_features=16,
        context_features=2,
        num_blocks=2,
        patch_size=2,
        normlayer=None # nn.LayerNorm
    )

    transform.forward(inputs, context)
    transform.inverse(inputs, context)


if __name__ == '__main__':
    main()