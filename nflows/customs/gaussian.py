
from typing import Any, Callable, List, Mapping, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


## ==============================
class Gaussian(nn.Module):
    def __init__(self, 
                 shape: Union[List, Tuple],
                 mu: float=0.0,
                 sigma: float=1.0) -> None:
        super().__init__()

        self._shape = torch.Size(shape)

        self.register_buffer('_mu', torch.tensor(mu, dtype=torch.float32))
        self.register_buffer('_sigma', torch.tensor(sigma, dtype=torch.float32))

        self.register_buffer('_log_p_const', torch.tensor(
                             np.log(1 / (2 * torch.pi)**0.5).astype(np.float32)))


    # -----
    def sample(self, num_samples:int, context:Tensor=None) -> Tensor:
        """Generates samples from the distribution. Samples can be generated in batches.

        Params:
        ----
        num_samples : int
            Number of samples
        context : Tensor
            `[B, context_dim]`, conditioning variables
        
        Returns:
        ----
        samples : Tensor
            `[num_samples, _shape]` if context is None, or `[B, num_samples, _shape]` if 
            context is given.

        """
        if context is None:
            samples = torch.normal(mean=self._mu, std=self._sigma, 
                                   size=(num_samples, *self._shape),
                                   device=self._log_p_const.device)
        else:
            B = context.shape[0]
            samples = torch.normal(mean=self._mu, std=self._sigma, 
                                   size=(B, num_samples, *self._shape),
                                   device=self._log_p_const.device)
        return samples



    # -----
    def log_prob(self, inputs:Tensor, context:Tensor=None) -> Tensor:
        """Calculate log probability under the distribution.

        Params:
        ----
        inputs : Tensor
            `[N, self._shape]`, input sample with shape of [batch_size, self._shape]
        context : Tensor
            Always None here.
        
        Returns:
        ----
        log_prob : Tensor
            `[N]`, the log probability of the inputs with same length as inputs.

        """
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:])
                )

        log_prob_no_grad_part = (self._log_p_const + torch.log(1 / self._sigma))
        log_prob = log_prob_no_grad_part \
                 - ((inputs - self._mu).square() / (2 * (self._sigma**2))) # [B, _shape]
        log_prob = log_prob.sum(dim=list(range(1, log_prob.ndim))) # [B,]
        
        return log_prob


    # -----
    def sample_and_log_prob(self, num_samples:int, context:Tensor=None) -> Tuple[Tensor, Tensor]:
        """Generates samples from the distribution together with their log probability.

        Params:
        ----
        num_samples : int
            Number of samples to generate.
        context : Tensor
            `[B, context_dim]`, conditioning variables. If None, the context is ignored.

        Returns:
        ----
        samples : Tensor
            `[num_samples, _shape]` if context is None, or `[B, num_samples, _shape]` if 
            context is given.
        log_prob : Tensor
            `[num_samples]` if context is None, or `[B, num_samples]` if context is given.

        """
        B = context.shape[0]
        samples = self.sample(num_samples, context=context)

        log_prob = self.log_prob(samples.reshape(-1, *self._shape), None)
        log_prob = log_prob.reshape(B, num_samples).contiguous()

        return samples, log_prob


    # -----
    @property
    def mean(self) -> Tensor:
        """ Reuturn the guassian's mean with 'self._shape'
        
        Returns:
        -------
        mean : Tensor
            `[_shape]`
        
        """
        return torch.zeros(self._shape, device=self._log_p_const.device) + self._mu.detach()


    # -----
    @property
    def std(self) -> Tensor:
        """ Reuturn the guassian's std with 'self._shape'

        Returns:
        -------
        mean : Tensor
            `[_shape]`
        
        """
        return torch.ones(self._shape, device=self._log_p_const.device) * self._sigma.detach()





if __name__ == '__main__':

    d = Gaussian(shape=[2, 3], mu=1.0, sigma=0.5).to(device='cuda:0')

    print(d.mean)
    print(d.std)