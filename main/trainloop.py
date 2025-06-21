
import os.path as osp
import pickle
import time
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.Arctic.dataset_frame import ArcticFrameDataset
from data.Dexycb.dataset_frame import DexycbFrameDataset
from data.Ho3dv3.dataset_frame import Ho3dv3FrameDataset, Ho3dv3MheFrameDataset
from tools.networkUtils import load_state_dict_partial
from tools.trainingUtils import (HalfCosineLrAdjuster, LogPrinter, LossBase,
                                 LossLogger)

from .loss import AmbHandLoss
from .model import AmbHand


# -----------------------------
def get_trainset_testset(data_name:str, data_dir:str, **kwargs):

    if data_name == 'Dexycb':
        trainset = DexycbFrameDataset(data_dir, mode='train', **kwargs)
        testset = DexycbFrameDataset(data_dir, mode='test', **kwargs)
        # trainset = DexycbFrameDataset(data_dir, mode='val', **kwargs) # fixme: for debug
        # testset = DexycbFrameDataset(data_dir, mode='val', **kwargs)
    elif data_name == 'HO3Dv3_MHE':
        trainset = Ho3dv3MheFrameDataset(data_dir, mode='train', **kwargs)
        testset = Ho3dv3MheFrameDataset(data_dir, mode='test', **kwargs)
    elif data_name == 'HO3Dv3':
        trainset = Ho3dv3FrameDataset(data_dir[0], mode='train', **kwargs)
        testset = DexycbFrameDataset(data_dir[1], mode='val', **kwargs)
    elif data_name == 'Arctic_P1':
        trainset = ArcticFrameDataset(data_dir, protocal='p1', mode='train', hand_type='right', decimation=4, **kwargs)
        testset = ArcticFrameDataset(data_dir, protocal='p1', mode='val', hand_type='right', decimation=4, **kwargs)
    elif data_name == 'Arctic_P2':
        trainset = ArcticFrameDataset(data_dir, protocal='p2', mode='train', hand_type='right', decimation=None, **kwargs)
        testset = ArcticFrameDataset(data_dir, protocal='p2', mode='val', hand_type='right', decimation=None, **kwargs)
    else:
        raise KeyError('Invliad Data Name.')

    return trainset, testset




# -----------------------------
def training_process(
        rank: int,
        num_gpus: int,

        seed: int=3407,
        batch_size: int=64,

        resolution: int=256,
        num_samples: int=257,

        which_stage: str='S1',
        S1_loss_weights: List=[...],
        S2_loss_weights: List=[...],

        num_epochs: int=40,
        num_warmup: int=5,
        max_lr: float=2.5e-4,
        min_lr: float=1e-6,

        resume_pkl: str=None,

        data_name:str='Dexycb',
        data_dir:str='Datasets/Dexycb',

        num_workers: int=6,
        run_dir: str='_rundir/xx',
        iters_per_log: int=200, # print log after every n train iters.
        **kwargs,
):

    device = torch.device('cuda', index=rank)
    process_seed = seed * num_gpus + rank
    torch.manual_seed(process_seed)
    torch.backends.cudnn.benchmark = True

    assert which_stage in ['S1', 'S2', 'All'] # all means train both stage.
    train_S1  = (which_stage == 'S1')
    train_S2  = (which_stage == 'S2')
    train_all = (which_stage == 'All')

    # ~~~~~~~~~~~~~~~~~~~~~~
    # 1. DATA PREPARING
    # ~~~~~~~~~~~~~~~~~~~~~~
    if rank == 0:
        log_printer = LogPrinter(osp.join(run_dir, 'log.txt'))
        log_printer.add('1. Initializing Dataset...', end='')

    trainset, testset = get_trainset_testset(data_name, data_dir, resolution=resolution, **kwargs)

    # same seed for sampler to avoid get same data in an epoch for all process.
    train_sampler = DistributedSampler(trainset, rank=rank, seed=seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    _iters_per_epoch_train = len(trainloader)

    test_sampler = DistributedSampler(testset, rank=rank, seed=seed, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler,
                            num_workers=num_workers-2, pin_memory=True, persistent_workers=True)
    _iters_per_epoch_test = len(testloader)

    if rank == 0:
        log_printer.add(f'{len(trainset)} train data and {len(testset)} test data.')


    # ~~~~~~~~~~~~~~~~~~~~~~
    # 2. MODEL INITIALIZATION
    # ~~~~~~~~~~~~~~~~~~~~~~
    if rank == 0:
        log_printer.add('2. Initializing DDP models and Optimizers...', end='')

    model = AmbHand(num_samples=num_samples,
                    frozen_backbone= (not train_S1) and (not train_all),
                    frozen_stage_1= (not train_S1) and (not train_all),
                    frozen_stage_2= (not train_S2) and (not train_all))

    if resume_pkl is not None:
        snapshot = pickle.load(open(resume_pkl, 'rb'))['snapshot']
        model = load_state_dict_partial(model, snapshot)

    model.to(device=device)
    model.train()
    net_name = model.name

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=min_lr, weight_decay=1e-4)
    lr_adjuster = HalfCosineLrAdjuster(base_lr=max_lr, min_lr=min_lr,
                                       num_warmup=num_warmup, num_epochs=num_epochs,
                                       iters_per_epoch=_iters_per_epoch_train)

    lossfn :LossBase = AmbHandLoss()

    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=True)
    model.eval()

    if rank == 0:
        log_printer.add(f'Done. Use {net_name} and resume from {resume_pkl}')


    ## ============================= Train Main ======================================
    if rank == 0:
        log_printer.add(f'3. Starting training {which_stage}...')
        logger = LossLogger(decimals=3, use_scientific=False)

    start_epoch = 0
    for epoch_counter in range(start_epoch, num_epochs):

        # ~~~~~~~~~~~~~~~~~~~~~~
        # 3. TRAINING PROCESS
        #~~~~~~~~~~~~~~~~~~~~~~
        model.train()
        train_sampler.set_epoch(epoch_counter)

        if rank == 0:
            logger.clear()
            lr = optimizer.state_dict()['param_groups'][-1]['lr']
            log_printer.add('-' * 30)
            log_printer.add(f'Training Epoch {epoch_counter+1}/{num_epochs} with initial lr {lr:.6f}:')

        iter_counter = 0
        for batch_data in trainloader:
            iter_counter += 1

            # train main
            for key in batch_data.keys(): # to cuda
                batch_data[key] = batch_data[key].to(device=device)
            batch_data = model.module.add_v3d_weakcam_for_gts(batch_data) # to generate gt_v3d and gt_weakcam

            optimizer.zero_grad()

            stage1_preds, stage2_preds = model(batch_data, train_S1)

            if train_S1:
                loss = lossfn.compute_loss_stage_1(stage1_preds, batch_data, S1_loss_weights)
            elif train_S2:
                loss = lossfn.compute_loss_stage_2(stage2_preds, batch_data, S2_loss_weights)
            elif train_all:
                loss = lossfn.compute_loss_stage_1(stage1_preds, batch_data, S1_loss_weights) \
                     + lossfn.compute_loss_stage_2(stage2_preds, batch_data, S2_loss_weights)
            else:
                ...
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2) # prevent grad explosion but slow the train speed.
            optimizer.step()

            lr_adjuster.adjust(epoch_counter, iter_counter, optimizer) # update lr.

            # logging
            if rank == 0:
                logger.append_new(lossfn.named_losses)
                if (iter_counter == 1) or \
                        (iter_counter % iters_per_log == 0) or \
                            (iter_counter == _iters_per_epoch_train):

                    _, display_log = logger.aggregate_stored_logs()
                    evolve = f'{iter_counter / _iters_per_epoch_train * 100:> 3.1f}%'.rjust(7)
                    nowtime = time.strftime('%H:%M:%S', time.localtime(time.time()))
                    log_printer.add(nowtime + ' ' + evolve + ' ', display_log)


        # ~~~~~~~~~~~~~~~~~~~~~~
        # 4. TESTING PROCESS
        # ~~~~~~~~~~~~~~~~~~~~~~
        model.eval()

        if rank == 0:
            log_printer.add(f'Testing Epoch {epoch_counter + 1} ...\n', end='')

        # test forward
        TestErrs = dict()
        with torch.no_grad():
            for batch_data in testloader:
                for key in batch_data.keys(): # to cuda
                    batch_data[key] = batch_data[key].to(device=device)
                batch_data = model.module.add_v3d_weakcam_for_gts(batch_data) # to generate gt_v3d and gt_weakcam

                stage1_preds, stage2_preds = model(batch_data, train_S1)

                if train_S1:
                    lossfn.compute_loss_stage_1(stage1_preds, batch_data, S1_loss_weights)
                elif train_S2:
                    lossfn.compute_loss_stage_2(stage2_preds, batch_data, S2_loss_weights)
                elif train_all:
                    lossfn.compute_loss_stage_1(stage1_preds, batch_data, S1_loss_weights)
                    lossfn.compute_loss_stage_2(stage2_preds, batch_data, S2_loss_weights)

                for key,val in lossfn.named_losses.items(): # to store the err value
                    if TestErrs.get(key, None) is not None:
                        TestErrs[key] += val
                    else:
                        TestErrs[key] = val

        # multi processes synchron.
        for key in TestErrs.keys():
            TestErrs[key] /= _iters_per_epoch_test
            TestErrs[key] /= num_gpus
            if num_gpus > 1:
                dist.all_reduce(tensor=TestErrs[key])
                dist.broadcast(TestErrs[key], src=0)

        if rank == 0:
            for key in TestErrs.keys():
                log_printer.add(f'Test {key}-err:{TestErrs[key].detach():<.2f}')

        # save model
        if rank == 0:
            model_snapshot = dict()
            for k in model.module.state_dict().keys(): # ddp(net).module == net
                model_snapshot.update({k:model.module.state_dict()[k].to(device='cpu')})

            mpjpe = TestErrs[f'S2_z0j3d'].item() if (train_S2 or train_all) else TestErrs[f'S1_z0j3d'].item()
            save_pkl = dict(progress=(epoch_counter+1) / num_epochs, snapshot=model_snapshot)
            save_name = osp.join(run_dir, f'Epoch_{epoch_counter+1:02d}_MPJPE_{mpjpe:2.2f}.pkl')
            pickle.dump(save_pkl, open(save_name, 'wb'))

            log_printer.add(f'Current Model Saved in {save_name}')
            log_printer.add('')

