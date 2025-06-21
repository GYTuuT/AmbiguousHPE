
import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm


## =====================
class MaximumMeanDiscrepancy(nn.Module):
    """ MMD distance between two sample sets.

    Refer to https://github.com/jindongwang/transferlearning/blob/master/code/DeepDA/loss_funcs/mmd.py

    """
    def __init__(self,
                 kernel_type: str='rbf',
                 kernel_mul: float=2.0,
                 kernel_num: int=5,
                 fix_sigma: float=None):
        super(MaximumMeanDiscrepancy, self).__init__()

        assert kernel_type in ['rbf', 'linear']

        self.kernel_type = kernel_type
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma


    # ------------------
    def guassian_kernel(self, source:Tensor, target:Tensor) -> Tensor:
        """
        source: [B, Ns, K], B batches of Ns source samepls with K features
        target: [B, Nt, K], B batches of Nt target samepls with K features
        """
        n_samples = int(source.size()[1]) + int(target.size()[1])

        total = torch.cat([source, target], dim=1)
        total0 = total.unsqueeze(2).expand(
            -1, total.size(1), total.size(1), total.size(2))
        total1 = total.unsqueeze(1).expand(
            -1, total.size(1), total.size(1), total.size(2))
        L2_distance = ((total0 - total1) ** 2).sum(-1)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = L2_distance.sum(dim=(1,2), keepdim=True) / (n_samples ** 2 - n_samples + 1e-8)

        bandwidth = (bandwidth /  self.kernel_mul ** (self.kernel_num // 2))[None, ...] # [1, B, n, n]
        bandwidth = bandwidth * torch.pow(self.kernel_mul,
            torch.arange(self.kernel_num, device=source.device).reshape(-1, 1, 1, 1))

        kernel = torch.exp(-L2_distance[None, ...] / bandwidth + 1e-8).sum(0)

        return kernel


    # -------------------
    def linear_mmd2(self, f_of_X:Tensor, f_of_Y:Tensor) -> Tensor:
        delta = f_of_X.mean(1) - f_of_Y.mean(1)
        loss = (delta ** 2).sum(1)

        return loss


    # ------------------
    def forward(self, source:torch.Tensor, target:torch.Tensor) -> Tensor:
        """
        source: [B, Ns, K], B batches of Ns source samepls with K features
        target: [B, Nt, K], B batches of Nt target samepls with K features

        return: [B,]
        """
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)

        elif self.kernel_type == 'rbf':
            kernels = self.guassian_kernel(source, target)
            N = source.size(1)
            reduce_dims = list(range(1, kernels.ndim))
            XX = kernels[:, :N, :N].mean(reduce_dims)
            YY = kernels[:, N:, N:].mean(reduce_dims)
            XY = kernels[:, :N, N:].mean(reduce_dims)
            YX = kernels[:, N:, :N].mean(reduce_dims)
            loss = XX + YY - XY - YX

            return loss

        else:
            raise ValueError("Unknown kernel type: %s" % self.kernel_type)



## ---------------------------
def get_PaEuc_batch(predicted, target, valid:np.ndarray=None, return_aligned:bool=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.

    Parameter:
    ----
    predicted: ndarray
        [B, N, 3], predicted points
    target: ndarray
        [B, N, 3 ], gt points value
    return_aligned: bool
        if return the aligned predicted back.
    """
    assert predicted.shape == target.shape
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY
    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t
    pampjpe = np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1)
    if valid is not None:
        pampjpe = np.mean(pampjpe[valid > 0])
    else:
        pampjpe = np.mean(pampjpe)

    # Return PA-MPJPE
    if return_aligned:
        return predicted_aligned, pampjpe
    else:
        return pampjpe


# ----------------------------
def compute_Joints_AUC(preds, gts, valid=None, range_min:float=0.0, range_max:float=50.0, steps=100):
    """ Compute the AUC within the range of [range_min, range_max]
    preds: [N, 21, 3], in millimeters
    gts: [N, 21, 3], In millimeters
    valid: [N,], the validilty of the hand. True for valid, False for invalid, default is None.
    """
    assert len(preds) == len(gts)
    assert range_min < range_max
    if valid is not None:
        assert len(valid) == len(preds), f"Invalid shape: {len(valid)}, need {len(preds)}"

    thresholds = np.linspace(range_min, range_max, steps) # [steps,]
    Pck = np.zeros((steps)) # [steps,]

    for idx in tqdm(range(len(preds))):
        if valid is not None:
            if not valid[idx]:
                continue

        ra_p = preds[idx] - preds[idx][:1] # [21, 3]
        ra_g = gts[idx] - gts[idx][:1] # [21, 3]

        e = np.sqrt(np.sum((ra_p - ra_g)**2, axis=-1)) # [21,]
        e = np.repeat(e[None,...], steps, axis=0) # [steps, 21]

        Pck += np.sum(e < thresholds[:, None], axis=-1) # [steps,]

    num_valid = np.sum(valid) if valid is not None else len(preds)
    Pck = Pck / (21 * num_valid) # [steps,]
    auc = Pck.mean()

    # print(f'AUC: {Pck.mean():0.3f}')

    return auc

# ----------------------------
def compute_Joint_Fscore(preds, gts, threshold:float=0.05, valid=None):
    """ Compute the F-score of the joints.
    preds: [N, 21, 3], in millimeters
    gts: [N, 21, 3], In millimeters
    threshold: float, the threshold of the distance between the predicted and gt joints.
    valid: [N,], the validilty of the hand. True for valid, False for invalid, default is None.
    """
    assert len(preds) == len(gts)
    if valid is not None:
        assert len(valid) == len(preds), f"Invalid shape: {len(valid)}, need {len(preds)}"

    num_valid = np.sum(valid) if valid is not None else len(preds)
    num_correct = 0
    num_predicted = 0
    num_gt = 0

    for idx in tqdm(range(len(preds))):
        if valid is not None:
            if not valid[idx]:
                continue

        ra_p = preds[idx] - preds[idx][:1] # [21, 3]
        ra_g = gts[idx] - gts[idx][:1] # [21, 3]

        e = np.sqrt(np.sum((ra_p - ra_g)**2, axis=-1)) # [21,]
        e = np.repeat(e[None,...], 21, axis=0) # [steps, 21]

        num_correct += np.sum(e < threshold)
        num_predicted += 21
        num_gt += 21

    precision = num_correct / num_predicted
    recall = num_correct / num_gt
    fscore = (2 * precision * recall) / (precision + recall)

    print(f'F-score: {fscore:0.3f}')

    return fscore
