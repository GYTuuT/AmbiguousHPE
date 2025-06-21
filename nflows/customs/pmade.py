
import torch
from torch import Tensor, nn
from torch.nn import functional as F


## ------------------------
def _create_degrees(n:int) -> Tensor:
    """
    Creates the 1D degrees with n features.
    """
    return torch.arange(1, n+1)


## ========================
class PatchMaskedLinear(nn.Linear):
    """
    PieceMaskedLinear is a subclass of PyTorch's nn.Linear class, representing a linear transformation to the incoming data. It is designed to handle input features, output features, and autoregressive features that are divided into patches of a specified size.
    """

    weight_mask : Tensor
    out_degrees : Tensor

    def __init__(self, 
            in_features: int, 
            out_features: int, 
            ar_features: int, # Autoregressive features
            patch_size: int=1,
            in_degrees: Tensor=None,
            is_output: bool=False,
            random_mask: bool=False,
            bias: bool = True) -> None:
        super().__init__(in_features=in_features, 
                         out_features=out_features, 
                         bias=bias)
        
        assert (in_features % patch_size == 0) \
            and (out_features % patch_size == 0) \
                and (ar_features % patch_size == 0), \
                    "Feature size should be multiple of patch_size"

        self.in_features  = in_features
        self.out_features = out_features
        self.ar_features  = ar_features
        self.patch_size = patch_size

        self.in_patchs  = in_features  // patch_size
        self.out_patchs = out_features // patch_size
        self.ar_patchs  = ar_features  // patch_size
    
        self.in_degrees = in_degrees if in_degrees is not None else _create_degrees(self.in_patchs)
        assert (len(self.in_degrees) * patch_size) == in_features, "Invalid Inputs Settings"

        mask, out_degrees = self._create_mask_and_degrees(is_output, random_mask)
        self.register_buffer('weight_mask', mask.to(dtype=torch.float32))
        self.register_buffer('out_degrees', out_degrees.to(dtype=torch.long))


    # ---------------
    def _create_mask_and_degrees(self, is_output:bool=False, random_mask:bool=False):
        if is_output:
            out_degrees = _create_degrees(self.ar_patchs)
            out_degrees = out_degrees.unsqueeze(-1).repeat(1, self.out_patchs // self.ar_patchs)
            out_degrees = out_degrees.reshape(-1)
            
            pmask = (out_degrees[..., None] > self.in_degrees) * 1.0
        
        else: 
            if random_mask:
                _min = min(self.in_degrees.min(), self.ar_patchs - 1)
                out_degrees = torch.randint(low=_min, high=self.ar_patchs, size=[self.out_patchs])
            
            else:
                _max, _min = max(1, self.ar_patchs - 1), min(1, self.ar_patchs - 1)
                out_degrees = torch.arange(self.out_patchs) % _max + _min

            pmask = (out_degrees[..., None] >= self.in_degrees) * 1.0

        mask = pmask[..., None, None].repeat(1, 1, self.patch_size, self.patch_size)
        mask = mask.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)

        out_degrees = out_degrees.to(dtype=torch.long)

        return mask, out_degrees
    

    # ----------------
    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight * self.weight_mask, self.bias)





## ================
class PatchMaskedResidualBlock(nn.Module):
    """
    A residual block with patch masking for autoregressive models.

    Args:
        features (int): The number of input and output features.
        ar_features (int): The number of autoregressive features.
        context_features (int, optional): The number of condition context features. Defaults to None.
        patch_size (int, optional): The size of the patch. Defaults to 1.
        in_degrees (Tensor, optional): The input degrees. Defaults to None.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.
        activation (nn.Module, optional): The activation function. Defaults to nn.LeakyReLU.
        normlayer (nn.Module, optional): The normalization layer. Defaults to nn.BatchNorm1d.

    Attributes:
        out_degrees (Tensor): The output degrees.

    """
    out_degrees : Tensor

    def __init__(self,
            features: int,
            ar_features: int, # autoregressive features
            context_features: int=None, # condition context features
            patch_size: int=1,
            in_degrees: Tensor=None,
            dropout_rate: float=0.,
            activation: nn.Module=nn.LeakyReLU,
            normlayer: nn.Module=nn.BatchNorm1d
            ) -> None:
        super().__init__()

        assert (features % patch_size == 0) \
                and (ar_features % patch_size == 0), \
                    "Feature size should be multiple of patch_size"
        
        self.in_degrees = _create_degrees(features // patch_size) if (in_degrees is None) else in_degrees
        assert (len(self.in_degrees) * patch_size) == features, "Invalid Inputs Settings"

        self.do_norm = (normlayer is not None)

        self.context_layer = None
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
            self.context_norm = normlayer(features) if self.do_norm else None

        self.layer_0 = PatchMaskedLinear(
            in_features=features,
            out_features=features,
            ar_features=ar_features,
            patch_size=patch_size,
            in_degrees=self.in_degrees,
            random_mask=False,
            is_output=False
        )

        self.layer_1 = PatchMaskedLinear(
            in_features=self.layer_0.out_features,
            out_features=features,
            ar_features=ar_features,
            patch_size=patch_size,
            in_degrees=self.layer_0.out_degrees,
            random_mask=False,
            is_output=False
        )

        self.normlayer_0 = normlayer(features) if self.do_norm else None
        self.normlayer_1 = normlayer(features) if self.do_norm else None

        self.activation = activation(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)

        self.out_degrees = self.layer_1.out_degrees
        assert (self.out_degrees >= in_degrees).all() == 1., "In a masked residual block, the output degrees can't be less than the corresponding input degrees."


    # ----------------
    def forward(self, inputs:Tensor, context:Tensor=None) -> Tensor: # connection likes ResNet's BasicBlock
        
        identity = inputs

        out = self.layer_0(inputs)
        out = self.normlayer_0(out) if self.do_norm else inputs
        
        if (context is not None) and (self.context_layer is not None):
            context_temps = self.context_layer(context)
            context_temps = self.context_norm(context_temps) if self.do_norm else context_temps

        out = self.activation(out + context_temps)
        out = self.dropout(out)

        out = self.layer_1(out)
        out = self.normlayer_1(out) if self.do_norm else out

        out = self.activation(out + identity) # residual connection.

        return out





## ===============
class PatchMADE(nn.Module):
    """
    PatchMADE is a type of autoregressive generative model that learns the probability distribution of a set of input data. It is based on the MADE model, but regress in patch step.
    """
    def __init__(self,
            in_features: int,
            hidden_features: int,
            context_features: int=None,
            patch_size: int=1,
            out_multiplier: int=1,
            num_blocks: int=2,

            dropout_rate: float=0.,
            activation: nn.Module=nn.LeakyReLU,
            normlayer: nn.Module=nn.BatchNorm1d,
            ) -> None:
        super().__init__()

        assert (in_features % patch_size == 0) \
            and (hidden_features % patch_size == 0), \
                    "Feature size should be multiple of patch_size"

        self.in_features = in_features
        self.in_patchs = in_features // patch_size
        self.in_degrees = _create_degrees(self.in_patchs)

        self.out_features = in_features * out_multiplier
        self.hidden_features = hidden_features

        self.do_norm = (normlayer is not None)

        # initial layer
        self.init_layer = PatchMaskedLinear(
            in_features=in_features,
            out_features=hidden_features,
            ar_features=in_features, # all the inputs features join the autoregression
            patch_size=patch_size,
            in_degrees=self.in_degrees,
            is_output=False
        )
        self.init_norm = normlayer(hidden_features) if self.do_norm else None
        self.activation = activation(inplace=False)

        self.context_layer = None
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)
            self.context_norm = normlayer(hidden_features) if self.do_norm else None

        # residual blocks
        residual_blocks = []
        prev_out_degrees = self.init_layer.out_degrees
        for _ in range(num_blocks):
            residual_blocks.append(
                PatchMaskedResidualBlock(
                    features=hidden_features,
                    ar_features=in_features,
                    context_features=context_features,
                    patch_size=patch_size,
                    in_degrees=prev_out_degrees,
                    dropout_rate=dropout_rate,
                    normlayer=normlayer,
                    activation=activation
                )
            )
            prev_out_degrees = residual_blocks[-1].out_degrees
        self.residual_blocks = nn.ModuleList(residual_blocks)

        # final layer
        self.final_layer = PatchMaskedLinear(
            in_features=hidden_features,
            out_features=self.out_features,
            ar_features=in_features,
            patch_size=patch_size,
            in_degrees=prev_out_degrees,
            is_output=True
        )

        self._init_weight()

    
    # --------------
    def _init_weight(self):

        nn.init.xavier_uniform_(self.init_layer.weight, gain=0.02)
        if self.context_layer is not None:
            nn.init.xavier_uniform_(self.context_layer.weight, gain=0.02)

        for block in self.residual_blocks:
            nn.init.xavier_uniform_(block.layer_0.weight, gain=0.02)
            nn.init.xavier_uniform_(block.layer_1.weight, gain=0.02)
            if block.context_layer is not None:
                nn.init.xavier_uniform_(block.context_layer.weight, gain=0.02)

        nn.init.xavier_uniform_(self.final_layer.weight, gain=0.02)


    # ---------------
    def forward(self, inputs:Tensor, context:Tensor=None) -> Tensor:
        
        out = self.init_layer(inputs)
        out = self.init_norm(out) if self.do_norm else out

        if (context is not None) and (self.context_layer is not None):
            context_temps = self.context_layer(context)
            context_temps = self.context_norm(context_temps) if self.do_norm else context_temps

        out = self.activation(out + context_temps)

        for res_block in self.residual_blocks:
            out = res_block(out, context)

        out = self.final_layer(out)

        return out







if __name__ == '__main__':

    # ## 1.PieceMaskedLinear Test
    # nin = 6
    # nout = 12
    # ps = 2

    # pl0 = PatchMaskedLinear(
    #         in_features=nin,
    #         out_features=24,
    #         ar_features=6,
    #         patch_size=ps,
    #         is_output=False)
    # pl1 = PatchMaskedLinear(
    #         in_features=pl0.out_features,
    #         out_features=nout,
    #         ar_features=6,
    #         patch_size=ps,
    #         in_degrees=pl0.out_degrees,
    #         is_output=True)
    
    # x = torch.tensor([[ 1.3796,  0.6631,  0.7048, -0.7281,  1.2404, -2.0317]])
    # y = torch.zeros_like(x)
    # for i in range(nin // ps + 1): # to check autoregression order.
    #     y = pl1(pl0(x))
    #     x = y.reshape(1, nin, nout//nin)[..., 0] * 0.353
    #     print(x)
    

    # exit()


    ## 2.PatchMADE Test
    nin = 8
    nout = 16
    ps = 2
    nc = 12

    pmade = PatchMADE(
        in_features=nin,
        hidden_features=1024,
        context_features=nc,
        patch_size=ps,
        out_multiplier=nout//nin,
        normlayer=None # nn.BatchNorm1d,
    )

    x = torch.randn([2, nin])
    c = torch.randn([2, nc])
    y = torch.zeros_like(x)

    for i in range(nin // ps + 1): # to check autoregression order.
        z = pmade(y, c)
        z = z.reshape(2, nin, nout//nin)
        y = z[..., 0] * x + z[..., 1] * x

        print(y[0])
    
    # y.mean().backward()

    # ## ----------------------
    # def check_nan_grad(module:nn.Module, show_nan:bool=True):
    #     params = module.named_parameters()
    #     for key, param in params:
    #         if param.grad is None:
    #             continue
    #         if torch.any(torch.isnan(param.grad)):
    #             if show_nan:
    #                 print(key)
    #             param.grad[param.grad!=param.grad] = 0 # nan != nan.
    
    # check_nan_grad(pmade)