

import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable

from nflows.customs.gaussian import Gaussian
from nflows.customs.pautoregressive import \
    PatchMaskedAffineAutoregressiveTransform
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AdditiveCouplingTransform, PiecewiseQuadraticCouplingTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.normalization import ActNorm
from nflows.transforms.permutations import (RandomPermutation,
                                            ReversePermutation)


# =====================
class Glow(Flow):

    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,

        num_layers: int=4,
        num_blocks_per_layer: int=2,
        activation: Callable=F.leaky_relu,

        dropout_probability: float=0.5,
        batch_norm_within_layers: bool=True,
    ):

        coupling_constructor = AdditiveCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                context_features=context_features,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            layers.append(ActNorm(features=features))
            layers.append(LULinear(features=features))
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            mask *= -1
            layers.append(transform)

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=Gaussian([features], mu=0.0, sigma=1.0)
        )

        # initialize the `ActNorm` of Glow before using.
        self.initialize_self(features, context_features)


    def initialize_self(self, features:int, context_features:int=None):
        _channels = 2 # any int value that > 1
        with torch.no_grad():
            self.log_prob(
                inputs=torch.randn([_channels, features]),
                context=torch.randn([_channels, context_features]) if context_features is not None else None)




## ==================
class PMAFlow(Flow): # Patch Masked AutoRegressive Flow

    def __init__(
            self,
            features: int,
            hidden_features: int,
            context_features: int=None,
            patch_size: int=1,

            num_layers: int=4,
            num_blocks_per_layer: int=2,

            use_random_permutations=False,
        ):


        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                PatchMaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    patch_size=patch_size,
                    num_blocks=num_blocks_per_layer,
                    normlayer=nn.BatchNorm1d
                )
            )

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=Gaussian([features], mu=0.0, sigma=1.0)
        )
