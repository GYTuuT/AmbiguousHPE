
import os
import sys

sys.path.append(os.getcwd())

import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.Dexycb.dataset_frame import DexycbFrameDataset
from data.Dexycb.evaluate import eval_Dexycb_Joints_AUC, eval_Dexycb_Mpjpe
from main.model import AmbHand
from main.modules.Manolayer import RotMatrixManolayer

# ----------
torch.manual_seed(0)
device = torch.device('cuda', 0)


# ----------
@torch.no_grad()
def get_AmbDexycb_Predictions(model, stage:str='S1', resolution:int=256):

    model.to(device=device)

    manolayer = RotMatrixManolayer('checkpoints/processed_mano', hand_type='right').to(device=device)
    testset = DexycbFrameDataset(data_dir='/root/Workspace/DATASETS/DexYCB', mode='test', flip_left_hand=True)
    testloader = DataLoader(testset, batch_size=64, num_workers=6, drop_last=False, shuffle=False)

    print('Model Predicting ...', end=' ')

    Pred_Joints3D = []
    Pred_HmJoints2D = []
    Pred_Roots3D = []
    Pred_Roots2D = []

    counter = 0
    for batch_data in tqdm(testloader):
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(device=device)

        # 2.0 Get predictions
        S1_preds, S2_preds = model(batch_data, only_stage_1=True if stage=='S1' else False)
        pd_hm_j2d = S1_preds['hm_j2d']

        if stage == 'S1':
            pd_z0_rmt = S1_preds['z0_rmt']
            pd_z0_shape = S1_preds['z0_shape']
            pd_z0_j2d = S1_preds['z0_j2d']
        else:
            pd_z0_rmt = S2_preds['z0_rmt'] # [B, 16, 3, 3]
            pd_z0_shape = S2_preds['z0_shape']
            pd_z0_j2d = S2_preds['z0_j2d']

        B = pd_z0_rmt.shape[0]

        # 2.2 Process predictions
        pd_z0_rmt = pd_z0_rmt.reshape(B, 16, 3, 3) # [B, 16, 3, 3]
        pd_z0_shape = pd_z0_shape.reshape(B, 10)
        pd_z0_v3d, pd_z0_j3d = manolayer(pd_z0_rmt[:, :1], pd_z0_rmt[:, 1:], pd_z0_shape) # [B, 778, 3], [B, 21, 3]

        for i in range(B):
            Pred_Joints3D.append(pd_z0_j3d[i].cpu().numpy())
            Pred_HmJoints2D.append(pd_hm_j2d[i].cpu().numpy())
            Pred_Roots3D.append(pd_z0_j3d[i][0].cpu().numpy())
            Pred_Roots2D.append(pd_z0_j2d[i][0].cpu().numpy())

        counter += B

    return Pred_Joints3D, Pred_HmJoints2D, Pred_Roots3D, Pred_Roots2D



# ---------------
def main(snapshot_name:str, stage:str):

    snapshot = pickle.load(open(snapshot_name, 'rb'))['snapshot']
    num_samples = 201

    model = AmbHand(num_samples=num_samples)
    model.load_state_dict(snapshot)
    model.eval()

    pred_j3ds, pred_hmj2ds, pred_roots3d, pred_roots2d = get_AmbDexycb_Predictions(model, stage)

    # saves = dict(
    #     pred_j3ds=np.stack(pred_j3ds, axis=0),
    #     pred_hmj2ds=np.stack(pred_hmj2ds, axis=0),
    #     pred_roots3d=np.stack(pred_roots3d, axis=0),
    #     pred_roots2d=np.stack(pred_roots2d, axis=0)
    # )
    # pickle.dump(saves, open('DexYCB_{}_preds.pkl'.format(stage), 'wb'))

    eval_Dexycb_Mpjpe(pred_j3ds, flip_left=True)
    eval_Dexycb_Joints_AUC(pred_j3ds, flip_left=True)



if __name__ == '__main__':

    main(snapshot_name='checkpoints/snapshots/DexYCB_JPE_11.68_wo_multiHypo.pkl', stage='S2')