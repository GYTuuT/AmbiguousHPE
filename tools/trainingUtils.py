
import functools
import math
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


## ----------------------
def check_nan_inf_grad(module:nn.Module, show_nan_inf:bool=True):
    params = module.named_parameters()
    for key, param in params:
        if param.grad is None:
            continue
        if torch.any(torch.isnan(param.grad) | torch.isinf(param.grad)):
            if show_nan_inf:
                print(key)
            param.grad[param.grad!=param.grad] = 0 # nan != nan.


## ===================
class HalfCosineLrAdjuster: # following MaskedAutoencoder
    def __init__(self,
                 base_lr:float,     # the base learning rate
                 min_lr:float,      # the minimun learning rate
                 num_warmup:int,    # num of warmup epochs
                 num_epochs:int,    # num of train epochs
                 iters_per_epoch:int,
                 **kwargs) -> None:
        
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.num_warmup = num_warmup
        self.num_epochs = num_epochs
        self.iters_per_epoch = iters_per_epoch

    ## -------------
    def get_current_lr(self, progress:float):
        
        if progress < self.num_warmup:
            lr = self.base_lr * progress / self.num_warmup
        else:
            scale = (progress - self.num_warmup) / (self.num_epochs - self.num_warmup)
            scale = 1.0 + math.cos(math.pi * scale)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * scale
        
        return float(lr)


    ## -------------
    def adjust(self, epoch_idx:int, iter_idx:int, optimizer:Optimizer=None):

        progress = iter_idx / (self.iters_per_epoch) + epoch_idx
        lr = self.get_current_lr(progress)

        # apply lr
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                if 'lr_scale' in param_group:
                    param_group['lr'] = lr * param_group['lr_scale']
                else:
                    param_group['lr'] = lr

        return lr



## =========================
class LossBase:
    """
    This is a base class for defining loss fn. It provides a mechanism to register and compute different types of sub losses.
    """

    RegisterLoss : Dict
    def __init__(self, 
                 basename:str=None, 
                 **kwargs) -> None:
        self.RegisterLoss = dict()
        self.basename = basename

    # -------------
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.compute_loss(*args, **kwds)

    # -------------
    def register_loss(name:str='.'):
        """
        Decorator for sub loss fun. To record the sub losses value with their names.

        Examples:

        .. code-block:: python
            # define
            @register_loss('nll')
            def subloss_fn_template(self, x, y, z):
                return x+y*z
            # when using
            .subloss_fn_template(1,2,3)
            print(.named_losses)
            # output
            {'nll': 5}

        """
        assert isinstance(name, str)
        def decorator(func:Callable):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                val = func(self, *args, **kwargs)
                if self.basename is not None:
                    self.RegisterLoss[self.basename + '_' + name] = val
                else:
                    self.RegisterLoss[name] = val
                return val
            return wrapper
        return decorator

    # --------------
    def compute_loss(self, *args: Any, **kwds: Any):
        """
        Main forward pass.
        """
        raise NotImplementedError
    
    # -------------
    @property
    def named_losses(self):
        """
        The computed loss store in the class.
        """
        return self.RegisterLoss



## =========================
class LossLogger:
    def __init__(self, decimals:int=3, use_scientific:bool=False) -> None:    
        self.recorder:Dict[List[np.ndarray]] = dict()
        self.decimals = decimals
        self.use_FE = use_scientific # if use scientific notation.

    ## -----
    def append_new(self, new_loss_dict: Dict):
        """
        Append new losses
        """
        for key in new_loss_dict.keys():

            value = new_loss_dict[key]
            if isinstance(value, torch.Tensor):
                value = value.clone().detach().to('cpu').numpy()

            if self.recorder.get('key', None) is None:
                self.recorder.update({key:[value]})
            else:
                self.recorder[key].append(value)

    ## -----
    def aggregate_stored_logs(self, **kwargs):
        """
        Aggregate the stored loss values by average their value.
        """
        mean_dict = dict()
        log_string = '| '
        for key in self.recorder.keys():
            value_list = self.recorder[key]
            try:
                value = np.array(value_list).mean()
                mean_dict.update({key:value})

                if not self.use_FE:
                    format_value = f'{value:0.{self.decimals}f}'[:self.decimals+2] # max string length, 0.000
                else:
                    format_value = f'{value:0.{self.decimals}e}'
                log_string = log_string + key + f': {format_value},  '

            except:
                mean_dict.update({key:None})
                log_string = log_string + key + ': None,  '
        
        return mean_dict, log_string

    ## -----
    def clear(self):
        """
        Clear the stored loss values.
        """
        self.recorder.clear()



## ===========
class LogPrinter:
    def __init__(self, log_file: str='',**kwargs) -> None:
        self.log_io = open(log_file, 'a')
        
    def add(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.log_io)
        self.log_io.flush()
