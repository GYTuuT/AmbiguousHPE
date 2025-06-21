
import os
import sys

sys.path.append(os.getcwd())

import json
import os.path as osp
import pickle
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from main.modules.Manolayer import RotMatrixManolayer
from tools.dataProcessing import apply_crop_resize
from datetime import datetime

#

## ---------------------
SWAP_MAT = np.array([[1.,  0.,  0.],
                     [0., -1.,  0.],
                     [0.,  0., -1.]], dtype=np.float32) # opengl -> camera.
SWAP_MAT_INV = np.linalg.inv(SWAP_MAT)

JOINT_REORDERS = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]



## ==================
class Ho3dv3EvalDataset(Dataset):
    def __init__(self, data_dir:str, resolution:int=256):
        self.data_dir = data_dir
        self.resolution = resolution

        with open(osp.join(data_dir, 'evaluation.txt')) as txtfile:
            _content = txtfile.readlines()
        self.filelist = [line.strip() for line in _content]

    # -------------
    def __len__(self):
        return len(self.filelist)

    # -------------
    def __getitem__(self, idx:int):
        filename = self.filelist[idx]
        frame_seq, frame_idx = filename.split('/')
        image_name = osp.join(self.data_dir, 'evaluation', frame_seq, 'rgb',  f'{frame_idx}.jpg')
        annos_name = osp.join(self.data_dir, 'evaluation', frame_seq, 'meta', f'{frame_idx}.pkl')

        annos = pickle.load(open(annos_name, mode='rb'))
        bbox = np.array(annos['handBoundingBox']) # top-left, bottom-right
        root = annos['handJoints3D'] @ SWAP_MAT # to camera space

        bbox_center = (bbox[:2] + bbox[2:]) / 2.
        edge = np.max(bbox[2:] - bbox[:2]) * 1.15 # same expansion as train dataset
        topleft = bbox_center - 0.5 * edge
        bbox = np.array([*topleft, edge, edge]).astype(np.float32)

        img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _ = apply_crop_resize(img, bbox, dst_size=(self.resolution, self.resolution))
        img = (img / 255.).astype(np.float32)

        return {'imgs':img, 'roots':root}



## -----------------------
@torch.no_grad()
def generate_ho3d_eval_zip(model:nn.Module, data_dir:str,
                           device:str=torch.device('cuda', 3)):
    batch_size = 64
    eval_data = Ho3dv3EvalDataset(data_dir, resolution=256)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=3)

    assert len(eval_data) == 20137, ValueError('Wrong evaluation data.')

    manolayer = RotMatrixManolayer(model_path='mano/mano_models/processed_mano',
                                   hand_type='right').to(device=device)

    model.to(device=device)
    model.eval()

    pred_Js = []
    pred_Vs = []
    for batch_data in tqdm(eval_loader):

        images = batch_data['imgs'].permute(0, 3, 1, 2).to(device=device) # [B, c, h, w]
        roots = batch_data['roots'].numpy()

        _, S2_preds = model({'image':images})

        z0_rmt = S2_preds['z0_rmt'] # [B, 16, 3, 3] # fixme:
        z0_shape = S2_preds['z0_shape'] # [B, 10]

        z0_v, z0_j = manolayer(z0_rmt[:, :1], z0_rmt[:, 1:],
                               z0_shape)
        z0_v = z0_v.cpu().numpy() # [B, 778, 3]
        z0_j = z0_j.cpu().numpy() # [B, 21, 3]

        z0_v = (z0_v - z0_j[:, :1] + roots.reshape(-1, 1, 3)) @ SWAP_MAT_INV # root align, camera -> opengl
        z0_j = (z0_j - z0_j[:, :1] + roots.reshape(-1, 1, 3)) @ SWAP_MAT_INV
        z0_j = z0_j[:, JOINT_REORDERS]

        pred_Js.extend(z0_j.tolist())
        pred_Vs.extend(z0_v.tolist())

    json_name = 'ho3dv3_eval.json'
    json.dump([pred_Js, pred_Vs], open(json_name, 'w'))

    data_tag = datetime.now().strftime('%m%d%H%M')
    os.system(f'zip -o -j HO3Dv3_eval_{data_tag}.zip {json_name}') # note: zip -j file.zip  file.json
    os.system(f'rm {json_name}')

    print(f'Dump {len(pred_Js)} J and {len(pred_Vs)} V predictions.')




if __name__ == '__main__':

    from main.model import AmbHand

    snapshot_name = '/root/Workspace/AmbHandNet_0309/checkpoints/snapshot_0320/HO3Dv3/S2.pkl'
    snapshot = pickle.load(open(snapshot_name, 'rb'))['snapshot']
    num_samples = 201

    model = AmbHand(num_samples=num_samples)
    model.load_state_dict(snapshot)

    generate_ho3d_eval_zip(model=model,
                           data_dir='/root/Workspace/DATASETS/HO3D_v3')



