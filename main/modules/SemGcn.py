
"""
Codes reference: https://github.com/garyzhao/SemGCN, Semantic Graph Convolutional Networks for 3D Human Pose Regression (CVPR 2019)
"""


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules import BatchNorm1d, LeakyReLU, Module


# --------------
def get_hand_skeleton(only_finger:bool=False):
    skeleton = [[0,  1], [ 1,  2], [ 2,  3], [ 3,  4],    # thumb
                [0,  5], [ 5,  6], [ 6,  7], [ 7,  8],    # index
                [0,  9], [9 , 10], [10, 11], [11, 12],    # middle
                [0, 13], [13, 14], [14, 15], [15, 16],    # ring
                [0, 17], [17, 18], [18, 19], [19, 20],]   # peaky (little)
    skeleton = torch.tensor(skeleton, dtype=torch.long)

    if only_finger:
        skeleton = skeleton[1,  2,  3,
                            5,  6,  7,
                            9, 10, 11,
                            13,14, 15]

    return skeleton


# --------------
def get_smpl_body_skeleton():
    skeleton = [[ 0,  1], [ 1,  4], [ 4,  7], [ 7, 10], # left leg
                [ 0,  2], [ 2,  5], [ 5,  8], [ 8, 11], # right leg
                [ 0,  3], [ 3,  6], [ 6,  9], [ 9, 12], [12, 15], # spine, coccyx -> neck
                [ 9, 13], [13, 16], [16, 18], [18, 20], [20, 22], # left arm
                [ 9, 14], [14, 17], [17, 19], [19, 21], [21, 23]] # right arm
    skeleton = torch.tensor(skeleton, dtype=torch.long)

    return skeleton


# ---------------
def get_AdjMat_from_edges(num_pts:int, edges:np.ndarray, symmetric_norm:bool=True):
    """Construct adjacency matrix from link edges
    Params:
        num_pts: int, number of graph points
        edges: Tensor, [N, 2], start and end point indices of linked edge.
        ymmetric_norm: if do symmetric normalized or not.
    Return:
        adj_mat: Tensor, [num_pts, num_pts]
    """

    adj = torch.zeros([num_pts, num_pts], dtype=torch.bool)
    adj[edges[:, 0], edges[:, 1]] = True
    adj = (adj.T | adj) * 1.0 # get symmetric adj matrix

    if symmetric_norm: # do Laplacian Symmetric normalized
        adj += torch.eye(*adj.shape) # A_hat
        deg = (adj.sum(dim=1) + 1e-8).diag() # D_hat
        # adj = deg.inverse() @ adj # (D_hat ** -1) @ A_hat
        adj = deg.inverse().sqrt() @ adj @ deg.inverse().sqrt() # (D_hat**-0.5) * A_hat * (D_hat**-0.5)

    return adj.to(dtype=torch.float32)



## =======================
class SemanticGraphConv(nn.Module):
    def __init__(self,
                 in_features:int,
                 out_features:int,
                 adj_matrix:Tensor,
                 bias:bool=True,
                 **kwargs) -> None:
        super().__init__()

        assert (adj_matrix.ndim == 2) and (adj_matrix.shape[0] == adj_matrix.shape[1]), \
            ValueError(f'Wrong shape {adj_matrix.shape} of input adj')
        # assert adj_matrix.diag().sum() > 1e-8, \
        #     ValueError('Diag of input adj should not be zero')

        self.in_features = in_features
        self.out_features = out_features

        # paramters
        _weight = torch.zeros([2, in_features, out_features], dtype=torch.float32) # two forward branch
        self.register_parameter('weight', nn.Parameter(_weight))

        _bias = torch.zeros(out_features, dtype=torch.float32) if bias else None
        self.register_parameter('bias', nn.Parameter(_bias) if _bias is not None else None)

        _learnable_adj_weight = torch.zeros([len(adj_matrix.nonzero())], dtype=torch.float32)
        self.register_parameter('adj_sparse_weight', nn.Parameter(_learnable_adj_weight))


        # buffers
        self.register_buffer('adj_matrix', adj_matrix.to(dtype=torch.float32))

        self.register_buffer('adj_nonzero_indices', (adj_matrix > 1e-8).to(dtype=torch.bool)) # to index the location of adj with valid connection.

        _self_connect_mask = torch.eye(*adj_matrix.shape, dtype=torch.float32)
        self.register_buffer('connect_mask', _self_connect_mask) # to split the self-connect part and cross connect part.


        self._init_param_()


    # ---------------
    def _init_param_(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.data.size(2))
            self.bias.data.uniform_(-stdv, stdv)

        nn.init.constant_(self.adj_sparse_weight.data, 1)


    # ---------------
    def forward(self, inputs:Tensor): # [B, N, features]

        A = -9e15 * torch.ones_like(self.adj_matrix)
        A[self.adj_nonzero_indices] = self.adj_sparse_weight
        A = F.softmax(A, dim=1) # normalized adj mat

        # self connect + cross connect
        outputs = (A * self.connect_mask) @ (inputs @ self.weight[0]) \
                + (A * (1 - self.connect_mask)) @ (inputs @ self.weight[1])

        if self.bias is not None:
            outputs = outputs + self.bias.view(1, 1, -1)

        return outputs


    def __repr__(self):
            return self.__class__.__name__ + \
                ' (' + str(self.in_features) + \
                    ' -> ' + str(self.out_features) + ')'



## ================
class ResSemanticGcnBlock(nn.Module): # be like Resnet BasicBlock
    def __init__(self,
                 features:int,
                 adj_matrix:Tensor,
                 activation: nn.Module=nn.LeakyReLU,
                 normlayer: nn.Module=nn.BatchNorm1d,
                 dropout: float=0.0,
                 **kwargs) -> None:
        super().__init__()

        self.features = features

        self.gcn1 = SemanticGraphConv(features, features, adj_matrix)
        self.norm1 = normlayer(features)
        self.activation = activation(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.gcn2 = SemanticGraphConv(features, features, adj_matrix)
        self.norm2 = normlayer(features)


    # -----------
    def forward(self, inputs:Tensor):

        identity = inputs

        out = self.gcn1(inputs).transpose(-1, -2) # [B, J, C] -> [B, C, J]
        out = self.norm1(out).transpose(-1, -2) # norm in (B,F), [B, C, J] -> [B, J, C]
        out = self.activation(out)

        out = self.dropout(out)

        out = self.gcn2(out).transpose(-1, -2)
        out = self.norm2(out).transpose(-1, -2)
        out = self.activation(out + identity)

        return out


## ==================
class ResSemanticGCN(nn.Module):
    def  __init__(self,
                  in_features:int,
                  hidden_features:int,
                  out_features:int,
                  adj_matrix:Tensor,
                  num_blocks:int=4,
                  activation: nn.Module=nn.LeakyReLU,
                  normlayer: nn.Module=nn.BatchNorm1d,
                  **kwargs) -> None:
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # input layer
        self.in_conv = SemanticGraphConv(in_features, hidden_features, adj_matrix)
        self.in_norm = normlayer(hidden_features)
        self.in_activation = activation(inplace=True)

        # res layers
        res_layers = [ResSemanticGcnBlock(hidden_features, adj_matrix,
                                          activation, normlayer)
                      for _ in range(num_blocks)]
        self.res_layers = nn.Sequential(*res_layers)

        # output layer
        self.out_conv = SemanticGraphConv(hidden_features, out_features, adj_matrix)


    # ---------
    def forward(self, inputs:Tensor): # [B, J, C]

        out = self.in_conv(inputs).transpose(-1, -2)
        out = self.in_norm(out).transpose(-1, -2)
        out = self.in_activation(out)

        out = self.res_layers(out)

        out = self.out_conv(out)

        return out

