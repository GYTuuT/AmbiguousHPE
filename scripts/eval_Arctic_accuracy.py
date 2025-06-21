import os
import sys

sys.path.append(os.getcwd())

import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.Arctic.dataset_frame import ArcticFrameDataset
from main.model import AmbHand
from main.modules.Manolayer import RotMatrixManolayer
from tools.metrics import get_PaEuc_batch, compute_Joints_AUC


# -------------------
device = torch.device('cuda',0)

# pretrained model
snapshot_name = 'checkpoints/snapshots/ARCTIC_P1_JPE_15.04_wo_multiHypo.pkl'
num_samples = 20

# data settings
protocal = 'p1'
hand_type = 'right'
decimation_interval = 4 if protocal == 'p1' else None

snapshot = pickle.load(open(snapshot_name, 'rb'))['snapshot']
model = AmbHand(num_samples=num_samples)
model.load_state_dict(snapshot)
model.to(device=device)
model.eval()

manolayer = RotMatrixManolayer('checkpoints/processed_mano', hand_type='right').to(device=device)
manolayer.to(device=device)

testset = ArcticFrameDataset(data_dir='/root/Workspace/DATASETS/ARCTIC', mode='val',
                             protocal=protocal, hand_type=hand_type, decimation_interval=decimation_interval)
testloader = DataLoader(testset, batch_size=64, num_workers=6, drop_last=False, shuffle=False)

# -------------------
print('Model Predicting ...', end=' ')

Pred_J3D = np.zeros([len(testset), 21, 3])
Real_J3D = np.zeros([len(testset), 21, 3])
Valid = np.zeros([len(testset)])
Real_Jvis = np.zeros([len(testset), 21]); Real_Jvis[:, 0] = 1
Pred_Inlrs = np.zeros([len(testset), 21]) # if inlier

counter = 0
for batch_data in tqdm(testloader):
    for key in batch_data.keys():
        batch_data[key] = batch_data[key].to(device=device)

    # 2.0 Get predictions
    with torch.no_grad():
        S1_preds, S2_preds = model(batch_data, only_stage_1=False)
    pd_hm_j2d = S1_preds['hm_j2d']
    pd_z0_rmt = S2_preds['z0_rmt'] # [B, 16, 3, 3]
    pd_z0_shape = S2_preds['z0_shape']
    pd_inlrs = S2_preds['inner_cfd']

    B = batch_data['is_right'].shape[0]

    # 2.2 Process predictions
    pd_z0_rmt = pd_z0_rmt.reshape(B, 16, 3, 3) # [B, 16, 3, 3]
    pd_z0_shape = pd_z0_shape.reshape(B, 10)
    pd_z0_v3d, pd_z0_j3d = manolayer(pd_z0_rmt[:, :1], pd_z0_rmt[:, 1:], pd_z0_shape) # [B, 778, 3], [B, 21, 3]

    Pred_J3D[counter:counter+B] = pd_z0_j3d.cpu().numpy()
    Real_J3D[counter:counter+B] = batch_data['joint3d'].cpu().numpy()
    Valid[counter:counter+B] = batch_data['hand_valid'].cpu().numpy()
    Real_Jvis[counter:counter+B, 1:] = batch_data['joint_vis'].cpu().numpy().reshape(B, 20)
    Pred_Inlrs[counter:counter+B] = pd_inlrs.cpu().numpy().reshape(B, 21)
    counter += B

# -------------------
Pred_J3D = (Pred_J3D - Pred_J3D[:, 0:1]) * 1000 # root aling
Real_J3D = (Real_J3D - Real_J3D[:, 0:1]) * 1000
mpjpe = np.linalg.norm(Pred_J3D - Real_J3D, axis=-1)
mpjpe = np.mean(mpjpe[Valid > 0])
pa_mpjpe = get_PaEuc_batch(Pred_J3D, Real_J3D, Valid)
auc = compute_Joints_AUC(Pred_J3D, Real_J3D, Valid)

print(f'Valid Rate: {np.mean(Valid): 0.3f}')
print(f'MPJPE: {mpjpe: 0.3f}')
print(f'PA-MPJPE: {pa_mpjpe: 0.3f}')
print(f'AUC: {auc: 0.3f}')






