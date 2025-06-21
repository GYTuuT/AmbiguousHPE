
import json
import os
import os.path as osp
from datetime import datetime
from tkinter import NO
from typing import Dict

import torch
import torch.distributed as dist

from main.trainloop import training_process
# from main_for_body.trainlopp_for_body import training_process_for_body # todo:
from tools.networkUtils import EasyDict as edict


# ---------
def subprocess_fn(rank:int, num_process:int, configs:Dict,
                  tcp_address:str='127.0.0.1:56789'):

    init_method = f'tcp://{tcp_address}' # can also be 'file://' or 'env://'
    dist.init_process_group(backend='nccl', rank=rank, world_size=num_process,
                            init_method=init_method)

    if configs.get('data_name', None) == 'HumanOH50K':
        pass
        # training_process_for_body(rank=rank, num_gpus=num_process, **configs)
    else:
        training_process(rank=rank, num_gpus=num_process, **configs)



# ---------
def Launch_Training(host:str, port:int, **configs):

    cfg = edict(**configs)

    if not hasattr(cfg, 'num_workers'):
        cfg.num_workers = 6
    if not hasattr(cfg, 'num_process'):
        cfg.num_process = 1
    if not hasattr(cfg, 'batch_size'):
        cfg.batch_size = 64
    if not hasattr(cfg, 'run_dir'):
        cfg.run_dir = osp.join(os.getcwd(), '_rundir')
        os.makedirs(cfg.run_dir, exist_ok=True)

    if hasattr(cfg, 'data_name'):
        data_name = cfg.data_name
    elif hasattr(cfg, 'data_dir'):
        data_name = osp.basename(cfg.data_dir)
    else:
        data_name = 'Unknown'

    cache_name = datetime.now().strftime('%m%d%H%M%S')
    cache_name = cache_name + f'_{data_name}'
    cfg.run_dir = osp.join(cfg.run_dir, cache_name)
    os.makedirs(cfg.run_dir, exist_ok=False)

    json.dump(cfg, open(osp.join(cfg.run_dir, 'TrainConfig.json'), 'w'), indent=4) # save configs

    tcp_address = f'{host}:{port}'
    print(f'Initializing Distributed Environments at {tcp_address}...')
    torch.multiprocessing.spawn(fn=subprocess_fn, nprocs=cfg.num_process,
                                args=(cfg.num_process, cfg, tcp_address))


S1_TRAINING_CONFIGS = dict(
    seed=3407, num_epochs=30, num_warmup=2, max_lr=2.5e-4, batch_size=64, num_process=2, num_samples=101, resolution=256,
    which_stage='S1',
    S1_loss_weights=[0.5, 0.3, 0.01], # nll, r6dreg, hm
    resume_pkl=None)

S2_TRAINING_CONFIGS = dict(
    seed=3407, num_epochs=20, num_warmup=2, max_lr=5e-5, batch_size=64, num_process=2, num_samples=101, resolution=256,
    which_stage='S2',
    S1_loss_weights=[0.2, 0.1, 0.01], # nll, r6dreg, hm
    S2_loss_weights=[0.5, 0.2, 0.1, 0.1, 0.1, 1.0], # nll, r6dreg, r6d, z0j3d,  visj2d, occstd/entropy
    resume_pkl=None)

DATASET_Cfgs = {
    'Dexycb':     dict(data_name='Dexycb',
                       data_dir='/root/Workspace/DATASETS/DexYCB'),
    'Arctic_P1':  dict(data_name='Arctic_P1',
                       data_dir='/root/Workspace/DATASETS/ARCTIC'),
    'Arctic_P2':  dict(data_name='Arctic_P2',
                       data_dir='/root/Workspace/DATASETS/ARCTIC'),
    'HO3Dv3_MHE': dict(data_name='HO3Dv3_MHE',
                       data_dir='/root/Workspace/DATASETS/HO3D_v3'),
    'HO3Dv3':     dict(data_name='HO3Dv3',
                       data_dir=['/root/Workspace/DATASETS/HO3D_v3', '/root/Workspace/DATASETS/DexYCB']),
    'HumanOH50K': dict(data_name='HumanOH50K',
                       data_dir='/root/Workspace/DATASETS/Human_OH50k'),
}



## ----------------------
if __name__ == '__main__':
    # note: 'mp.spawn' can only run after 'if __name__ == '__main__'' .
    # note: Pay attention to Random Seed, too large initial nll_loss will cause untrainable collapse.

    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    # config = S1_TRAINING_CONFIGS
    config = S2_TRAINING_CONFIGS
    config.update(DATASET_Cfgs['HO3Dv3_MHE'])

    Launch_Training(host='127.0.0.1', port=56789, **config)
