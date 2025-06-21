
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


## ===================
class PositionalEmbedding:
    # -----------
    @staticmethod
    def SinCos1D(num_token:int, dim_token:int, cls_token:bool=True) -> Tensor:
        # modified from https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/sine.py
        pos = ( # [N, D]
            torch.arange(0, num_token, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, dim_token)
        )
        dim = ( # [N, D]
            torch.arange(0, dim_token, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(num_token, 1)
        )
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / dim_token))

        pos *= div # [N+1, D]
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        if cls_token:
            cls_pos = torch.zeros([1, dim_token], dtype=torch.float32)
            pos = torch.cat([cls_pos, pos], dim=0) # [1+N, D]

        return pos
    
    # -------------
    @staticmethod
    def SinCos2D(num_token:int, dim_token:int, cls_token:bool=True) -> Tensor:
        # modified from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py#L20
        assert math.sqrt(num_token) % 1 == 0.0
        assert dim_token % 4 == 0.0

        h = w = int(math.sqrt(num_token))
        grid = torch.meshgrid(torch.arange(0, w, dtype=torch.float32),
                              torch.arange(0, h, dtype=torch.float32),
                              indexing='xy')
        grid = torch.stack(grid, dim=0).unsqueeze(dim=1) # 

        omega = torch.arange(0, dim_token // 4, dtype=torch.float32)
        omega = 1.0 / 10000**((omega * 2) / (dim_token / 2))

        emb_h = torch.einsum('k,d->kd', grid[0].reshape(-1), omega)
        emb_h = torch.cat([torch.sin(emb_h), torch.cos(emb_h)], dim=1)
        emb_w = torch.einsum('k,d->kd', grid[1].reshape(-1), omega)
        emb_w = torch.cat([torch.sin(emb_w), torch.cos(emb_w)], dim=1)
        pos = torch.cat([emb_h, emb_w], dim=1)

        if cls_token:
            cls_pos = torch.zeros([1, dim_token], dtype=torch.float32)
            pos = torch.cat([cls_pos, pos], dim=0) # [1+N, D]

        return pos



## =========================
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

    @staticmethod
    def drop_path(x, 
                  drop_prob: float = 0., 
                  training: bool = False, 
                  scale_by_keep: bool = True):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor



## =======================
class Mlp(nn.Module):
    def __init__(self, 
        in_features:int,
        hidden_features:int=None,
        out_features:int=None,

        act_layer: nn.Module=nn.GELU,
        mlp_drop: float=0.0,
        **kwargs) -> None:

        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, x:Tensor):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x 



## ====================
class SelfAttention(nn.Module):
    def __init__(self,
        dim:int,
        num_heads:int=8,
        qkv_bias:bool=False,
        qk_scale:float=None,

        attn_drop:float=0.0,
        proj_drop:float=0.0,
        **kwargs) -> None:

        super().__init__()

        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads

        self.scale = qk_scale or (head_dim ** -0.5)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x:Tensor):

        B, N, C = x.shape
        qkv = self.qkv(x) # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, n_heads, N, c_head]
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, n_heads, N, c_head]

        attn = (q @ k.transpose(-1,-2)) * self.scale # [B, n_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


## =====================
class SelfAttentionBlock(nn.Module):
    def __init__(self, 
        dim:int,
        drop_path:float=0.0,

        norm_layer:nn.Module=nn.LayerNorm,
        mlp_ratio:int=4,
        **kwargs) -> None: 

        super().__init__()
        
        self.norm_1 = norm_layer(dim)
        self.norm_2 = norm_layer(dim)
        self.attention = SelfAttention(dim, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(dim, int(dim*mlp_ratio), dim, **kwargs)

        self.apply(self._init_weights)

    # ---------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    # ---------
    def forward(self, x:Tensor): # [B, N, C], attention for N.

        x = x + self.drop_path(self.attention(self.norm_1(x))) # attn's residual skip
        x = x + self.drop_path(self.mlp(self.norm_2(x))) # mlp's residual skip

        return x



## ==================
class CrossAttention(nn.Module):
    def __init__(self,
        dim:int,
        num_heads:int=8,
        qkv_bias:bool=False,
        qk_scale:float=None,

        attn_drop:float=0.0,
        proj_drop:float=0.0,
        **kwargs) -> None:

        super().__init__()

        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads

        self.scale = qk_scale or (head_dim ** -0.5)
        
        self.q_linear  = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # ----------
    def forward(self, x:Tensor, y:Tensor): # x for q, y for kv
        
        B, N, C = x.shape
        q = self.q_linear(x).reshape(B, N, self.num_heads, C // self.num_heads)
        kv = self.kv_linear(y).reshape(B, N, 2, self.num_heads, C // self.num_heads)

        q, kv = q.permute(0, 2, 1, 3), kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # [B, heads, N, C_heads]
  
        attn = ((q @ k.transpose(-1, -2)) * self.scale).softmax(dim=-1) # [B, heads, N, N]
        attn = self.attn_drop(attn)

        y = (attn @ v).transpose(1, 2).reshape(B, N, C)
        y = self.proj_drop(self.proj(y))

        return y



## =====================
class CrossAttentionBlock(nn.Module):
    def __init__(self, 
        dim:int,
        drop_path:float=0.0,

        norm_layer:nn.Module=nn.LayerNorm,
        mlp_ratio:int=4,
        **kwargs) -> None: 

        super().__init__()
        
        self.norm_1 = norm_layer(dim)
        self.norm_2 = norm_layer(dim)
        self.attention = CrossAttention(dim, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(dim, int(dim*mlp_ratio), dim, **kwargs)

        self.apply(self._init_weights)

    # ---------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    # ---------
    def forward(self, x:Tensor, y:Tensor):  # x for q, y for kv
        # [B, N, C], attention for N.

        y = y + self.drop_path(self.norm_1(self.attention(x, y))) # attn's residual skip
        y = y + self.drop_path(self.norm_2(self.mlp(y))) # mlp's residual skip

        return y



