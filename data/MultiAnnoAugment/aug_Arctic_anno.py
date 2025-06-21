
import os
import sys

sys.path.append(os.getcwd())

import os.path as osp
import pickle
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.stats import truncnorm
from tqdm import tqdm

from data.Arctic.dataset import ArcticDataset
from data.MultiAnnoAugment.hand_status_check import (HandStatusCheck,
                                                     ObjectHandScene)
from main.modules.Manolayer import (Mano_Rotvec_Mean_Left,
                                    Mano_Rotvec_Mean_Right, ManoPcalayer,
                                    RotMatrixManolayer, RotVectorManolayer,
                                    convert_fingerOrder)
from tools.browserPloty import BrowserPlot
from tools.dataTransform import compute_bbox_M, produce_joint2d_bbox
from tools.networkUtils import gpu_running_timer
from tools.p3dRender import Pytorch3DRenderer, batch_project_K

# -------------
_img_size_p1 = (1000, 1000)
_bbox_size = (128, 128)


# =============
class ArcticDatasetForAugment(ArcticDataset):
    def __init__(self,
                 data_dir: str = '/root/Workspace/DATASETS/ARCTIC',
                 mode: str = 'val',
                 protocal: str = 'p1',
                 resolution: int = 256,
                 **kwargs) -> None:
        super().__init__(data_dir, mode, 'both', protocal, None, resolution, **kwargs)

    # -------------
    def __getitem__(self, idx: int) -> Dict:
        out = self.get_data(idx)
        for k in out.keys():
            if isinstance(out[k], np.ndarray):
                out[k] = torch.from_numpy(out[k])
        return out

    # -------------
    def get_data(self, data_idx: int, return_img:bool=True): # note: set False to avoid loading image, reduce time cost.

        index_chain = self.index_chains[data_idx]
        subject_id, sequence_name, view_idx, frame_idx = index_chain.split('.') # string

        image_name = osp.join(self.image_dir, index_chain.replace('.', '/') + '.jpg')
        image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB) if return_img else None

        anno_name = osp.join(self.anno_dir, index_chain.replace('.', '/') + '.pkl')
        anno = pickle.load(open(anno_name, 'rb'))

        cameraK = np.zeros([4, 4], dtype=np.float32)
        cameraK[:3, :3] = anno['cameraK'].reshape(3, 3)
        cameraK[2, 3] = cameraK[3, 2] = 1.0
        distort_coeffs = None if anno['distort_coeffs'] is None else anno['distort_coeffs'].reshape(8)

        mano_rotvec_r, mano_rotvec_l = anno['mano_pose_r'].reshape(16, 3), anno['mano_pose_l'].reshape(16, 3)
        mano_shape_r, mano_shape_l = anno['mano_shape_r'].reshape(10), anno['mano_shape_l'].reshape(10)
        mano_trans_r, mano_trans_l = anno['mano_trans_r'].reshape(3),  anno['mano_trans_l'].reshape(3)

        j2d_r, j2d_l = anno['j2d_r'].reshape(21, 2), anno['j2d_l'].reshape(21, 2)
        j3d_r, j3d_l = anno['j3d_r'].reshape(21, 3), anno['j3d_l'].reshape(21, 3)
        hand_valid_r, hand_valid_l = anno['hand_valid_r'], anno['hand_valid_l']
        if hand_valid_r and hand_valid_l:
            bbox = produce_joint2d_bbox(np.concatenate([j2d_r, j2d_l], axis=0), expansion=1.1)
        elif hand_valid_r and (not hand_valid_l):
            bbox = produce_joint2d_bbox(j2d_r, expansion=1.1)
        elif (not hand_valid_r) and hand_valid_l:
            bbox = produce_joint2d_bbox(j2d_l, expansion=1.1)
        else:
            bbox = None # NOTE: two hands are all invalid, means they aren't visible inside the image

        if bbox is not None:
            bbox_M = compute_bbox_M(bbox, dst_size=(self.resolution, self.resolution))
            projectK = cameraK.copy()
            projectK[:3, :3] = bbox_M @ projectK[:3, :3]
        else:
            projectK = None

        obj_verts, obj_faces = self.get_artic_object(anno['obj_name'],
                                                     anno['obj_top_world2cam_Rt'],
                                                     anno['obj_bottom_world2cam_Rt'])

        out = dict()
        out.update(dict(image=image)) if return_img else ...

        out.update(dict(obj_verts=obj_verts, obj_faces=obj_faces))

        out.update(dict(cameraK=cameraK, projectK=projectK, distort_coeffs=distort_coeffs, # [4,4], [8] or None

                        mano_rotvec_r=mano_rotvec_r, mano_shape_r=mano_shape_r, mano_trans_r=mano_trans_r, # [16,3], [10], [3]
                        mano_rotvec_l=mano_rotvec_l, mano_shape_l=mano_shape_l, mano_trans_l=mano_trans_l,

                        j2d_r=j2d_r, j2d_l=j2d_l, # [21,2], [21,2]
                        j3d_r=j3d_r, j3d_l=j3d_l, # [21,3], [21,3]
                        hand_valid_r=hand_valid_r, hand_valid_l=hand_valid_l, # bool, bool

                        bbox=bbox, # [4], [4]
                        ))
        return out

    # -------------
    @staticmethod
    def distort_pts(pts:torch.Tensor, coeffs:torch.Tensor):
        """
        Distort the 3D points with the given distortion coefficients.

        Args:
            pts (torch.Tensor): [B, N, 3]
            coeffs (torch.Tensor): [B, 8]
        """
        eps = 1e-8
        z = pts[..., 2]
        x1, y1 = pts[..., 0] / (z + eps), pts[..., 1] / (z + eps)
        x1_2, y1_2, x1_y1 = x1 * x1, y1 * y1, x1 * y1

        r2 = x1_2 + y1_2
        r4 = r2 * r2
        r6 = r4 * r2

        r_dist = (1 + coeffs[:, 0:1] * r2 + coeffs[:, 1:2] * r4 + coeffs[:, 4:5] * r6) \
               / (1 + coeffs[:, 5:6] * r2 + coeffs[:, 6:7] * r4 + coeffs[:, 7:8] * r6)

        # full (rational + tangential) distortion
        x2 = x1 * r_dist + 2 * coeffs[:, 2:3] * x1_y1 + coeffs[:, 3:4] * (r2 + 2 * x1_2)
        y2 = y1 * r_dist + 2 * coeffs[:, 3:4] * x1_y1 + coeffs[:, 2:3] * (r2 + 2 * y1_2)

        # denormalize for projection (which is a linear operation)
        distorted_pts = torch.stack([x2 * z, y2 * z, z], dim=-1)

        return distorted_pts




# -------------
def get_Arctic_visibility(mode:str='val', protocal:str='p1', device:str=torch.device('cuda', 2)) -> None:

    p3d_renderer = Pytorch3DRenderer(image_size=_bbox_size, device=device)
    manolayer = {
        'right':RotVectorManolayer(model_path='checkpoints/processed_mano', hand_type='right',
                                   finger_tips_mode='smplx').to(device=device),
        'left': RotVectorManolayer(model_path='checkpoints/processed_mano', hand_type='left',
                                   finger_tips_mode='smplx', fix_left_shapedirs=False).to(device=device)}
    dataset = ArcticDatasetForAugment('/root/Workspace/DATASETS/ARCTIC',
                                      mode=mode, protocal=protocal, resolution=_bbox_size[0])

    Jvis = np.zeros([len(dataset), 42])
    for data_idx in tqdm(range(len(dataset))):
        data_idx = random.randint(0, len(dataset)) # fixme: for debug
        data = dataset[data_idx]
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key][None].to(device=device) # add batch dim

        if (not data['hand_valid_l']) and (not data['hand_valid_r']): # see invalid hand as invisible hand.
            # print(f'Invalid hand in {data_idx}')
            continue

        hand_verts_r, hand_joints_r = manolayer['right'](data['mano_rotvec_r'][:, :1], data['mano_rotvec_r'][:, 1:],
                                                         data['mano_shape_r'], data['mano_trans_r'])
        hand_verts_l, hand_joints_l = manolayer['left' ](data['mano_rotvec_l'][:, :1], data['mano_rotvec_l'][:, 1:],
                                                         data['mano_shape_l'], data['mano_trans_l'])
        obj_verts, obj_faces = data['obj_verts'], data['obj_faces']

        if protocal == 'p2': # ego view should consider the camera distortion before projecting to the image plane.
            hand_verts_r = ArcticDatasetForAugment.distort_pts(hand_verts_r, data['distort_coeffs'])
            hand_verts_l = ArcticDatasetForAugment.distort_pts(hand_verts_l, data['distort_coeffs'])
            hand_joints_r = ArcticDatasetForAugment.distort_pts(hand_joints_r, data['distort_coeffs'])
            hand_joints_l = ArcticDatasetForAugment.distort_pts(hand_joints_l, data['distort_coeffs'])
            obj_verts = ArcticDatasetForAugment.distort_pts(obj_verts, data['distort_coeffs'])

        scene = ObjectHandScene(_img_size_p1, _bbox_size,
                                data['cameraK'], data['projectK'],
                                obj_verts * 1000, obj_faces,
                                hand_verts_l * 1000, manolayer['left'].faces[None],  hand_joints_l * 1000,
                                hand_verts_r * 1000, manolayer['right'].faces[None], hand_joints_r * 1000,)
        checker = HandStatusCheck(p3d_renderer, scene)

        joint_vis = checker.check_joint_visibility(return_mask=False) # [1, 42], no occluded
        joint_vis = joint_vis & checker.check_joint_outside_image().logical_not() # [1, 42], inside image
        joint_vis = joint_vis.float().squeeze()

        print('Left  Vis: \n', joint_vis[1:21].reshape(5, 4))
        print('Right Vis: \n', joint_vis[22: ].reshape(5, 4))

        nmap = checker.rendering_scene()
        brp = BrowserPlot(1, 2)
        brp.image(data['image'][0].cpu().numpy(), pos=[1,1], name=f'{data_idx}_image')
        brp.image(nmap[0].cpu().numpy(), pos=[1,2], name=f'{data_idx}_nmap')
        brp.show(61233)

        print()
        print(data_idx)
        print(data['hand_valid_l'], data['hand_valid_r'])
        exit()

        if not data['hand_valid_l']: # see invalid hand as invisible hand.
            joint_vis[:21] *= 0.0
        if not data['hand_valid_r']:
            joint_vis[21:] *= 0.0

        Jvis[data_idx] = joint_vis.cpu().numpy()

    Jvis = Jvis.astype(np.uint8)
    save_name = osp.join(dataset.data_dir, f'{dataset.mode}_{dataset.protocal}_visibility.pkl')
    pickle.dump(Jvis, open(save_name, 'wb'))



















# 2024.10.16, val_p1, 0 ~ 1/4., gpu 1
# 2024.10.16, val_p1, 1/4 ~ 1/2., gpu 0
# 2024.10.16, val_p1, 1/2 ~ 3/4., gpu 2
# 2024.10.16, val_p1, 3/4 ~ 1, gpu 3

# 2024.10.17, train_p1, 0~2.5w, gpu 0;
# 2024.10.17, train_p1, 2.5w~5w, gpu 1;
# 2024.10.17, train_p1, 5w~7.5w, gpu 2;
# 2024.10.17, train_p1, 7.5w~10w, gpu 3;

# 2024.10.17, train_p1, 10w~15w, gpu 0;
# 2024.10.17, train_p1, 15w~20w, gpu 1;
# 2024.10.17, train_p1, 20w~25w, gpu 2;
# 2024.10.17, train_p1, 25w~30w, gpu 3;

# 2024.10.18, train_p1, 30w~40w, gpu 0;
# 2024.10.18, train_p1, 40w~50w, gpu 1;
# 2024.10.18, train_p1, 50w~60w, gpu 2;
# 2024.10.18, train_p1, 60w~70w, gpu 3;

# 2024.10.19, train_p1, 70w~80w, gpu 0;
# 2024.10.19, train_p1, 80w~90w, gpu 1;
# 2024.10.19, train_p1, 90w~100w, gpu 2;
# 2024.10.19, train_p1, 100w~110w, gpu 3;

# 2024.10.21, train_p1, 110w~120w, gpu 0;
# 2024.10.21, train_p1, 120w~130w, gpu 1;
# 2024.10.21, train_p1, 130w~140w, gpu 2;
# 2024.10.21, train_p1, 140w~end, gpu 3;

# 2024.10.22, val_p2, all, gpu 1;

# 2024.10.23, train_p2, 0~10w, gpu 0;
# 2024.10.23, train_p2, 10w~20w, gpu 1;



# -------------
def augment_Arctic_multiAnno(mode:str='train', protocal:str='p1', device:str=torch.device('cuda', 1)) -> None:

    p3d_renderer = Pytorch3DRenderer(image_size=_bbox_size, device=device)
    manolayer = {
        'right':RotVectorManolayer(model_path='checkpoints/processed_mano', hand_type='right',
                                   finger_tips_mode='smplx').to(device=device),
        'left': RotVectorManolayer(model_path='checkpoints/processed_mano', hand_type='left',
                                   finger_tips_mode='smplx', fix_left_shapedirs=False).to(device=device)}
    pcalayer = ManoPcalayer(model_path='checkpoints/processed_mano').to(device=device)
    rotvec_means = {
        'right':torch.tensor(Mano_Rotvec_Mean_Right)[3:].to(device=device).reshape(-1, 45),
        'left':torch.tensor(Mano_Rotvec_Mean_Left)[3:].to(device=device).reshape(-1, 45),
    }
    dataset = ArcticDatasetForAugment('/root/Workspace/DATASETS/ARCTIC',
                                      mode=mode, protocal=protocal, resolution=_bbox_size[0])

    real_Jvis = pickle.load(open(osp.join(dataset.data_dir, f'{mode}_{protocal}_visibility.pkl'), 'rb'))
    real_Jvis = torch.from_numpy(real_Jvis.reshape(-1, 2, 21)).to(device=device) # [len_data, 2, 21], left hand and right fingers.

    # The standard deviation probability for the truncation range of the Gaussian sampler is set at 90%, thereby reducing the likelihood of obtaining unreasonable poses from PCA MANO due to extreme sampling.
    trunc_normal_sampler = truncnorm(-1.645, 1.645, loc=0, scale=1)
    num_random_per_finger = 100 # random poses for each invisible finger.

    for data_idx in tqdm(range(100000, len(dataset))):
        data = dataset[data_idx]
        multi_finger_poses = [None] * 10

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key][None,...].to(device=device)

        # 1. get visibliity information
        real_finger_joint_vis  = real_Jvis[data_idx][..., 1:].reshape(2, 5, 4) # [2, 20]
        real_finger_vis = real_finger_joint_vis.all(dim=-1) # see finger as visible if its all joints are visible.
        real_finger_vis_mano = convert_fingerOrder(real_finger_vis, 'nature2mano').reshape(10) # in MANO order, left hand first.

        # 2. generate random mano pose
        pca_randn = trunc_normal_sampler.rvs(size=[num_random_per_finger, 2, 40]) # 40 is the PCA components.
        pca_randn = torch.from_numpy(pca_randn).float().to(device=device) # [num_random_per_finger, 2, 40]
        rotvec_randn = torch.stack([
            pcalayer.pca2rotvec(pca_randn[:,0], 'left')  + rotvec_means['left'], # [num_random_per_finger, 45]
            pcalayer.pca2rotvec(pca_randn[:,1], 'right') + rotvec_means['right'], ], dim=1).reshape(-1, 10, 3, 3) # no root rotation.

        # 3. augmenting
        original_finger_poses = torch.cat([data['mano_rotvec_l'][:, 1:].reshape(-1, 5, 3, 3),
                                           data['mano_rotvec_r'][:, 1:].reshape(-1, 5, 3, 3)], dim=1) # [1, 10, 3, 3]

        if (not data['hand_valid_l']) and (not data['hand_valid_r']): # see invalid hand as invisible hand.
            multi_finger_poses = [rotvec_randn[:, finger_idx].cpu().numpy() for finger_idx in range(10)] # save 100 samples.
        else:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = torch.repeat_interleave(data[key], num_random_per_finger + 1, dim=0) # [num_random_per_finger + 1, ...]
            for finger_idx in range(10): # 2 hands, all 10 fingers.
                if real_finger_vis_mano[finger_idx]: # skip the augment of the fingers with no occluded joint.
                    multi_finger_poses[finger_idx] = original_finger_poses[:, finger_idx].cpu().numpy()
                    continue

                root_rotvec_l = data['mano_rotvec_l'][:, :1].clone().reshape(-1, 1, 3)
                finger_rotvec_l = data['mano_rotvec_l'][:, 1:].clone().reshape(-1, 5, 3, 3)
                root_rotvec_r = data['mano_rotvec_r'][:, :1].clone().reshape(-1, 1, 3)
                finger_rotvec_r = data['mano_rotvec_r'][:, 1:].clone().reshape(-1, 5, 3, 3)

                if finger_idx < 5: # replace the finger pose, and the first pose is the original pose.
                    finger_rotvec_l[1:, finger_idx] = rotvec_randn[:, finger_idx]
                else:
                    finger_rotvec_r[1:, finger_idx-5] = rotvec_randn[:, finger_idx]

                finger_rotvec_l = finger_rotvec_l.reshape(-1, 15, 3)
                finger_rotvec_r = finger_rotvec_r.reshape(-1, 15, 3)

                hand_verts_l, hand_joints_l = manolayer['left' ](root_rotvec_l, finger_rotvec_l,
                                                                data['mano_shape_l'], data['mano_trans_l'])
                hand_verts_r, hand_joints_r = manolayer['right'](root_rotvec_r, finger_rotvec_r,
                                                                data['mano_shape_r'], data['mano_trans_r'])
                obj_verts, obj_faces = data['obj_verts'], data['obj_faces']

                if protocal == 'p2': # ego view should consider the camera distortion before projecting to the image plane.
                    hand_verts_r = ArcticDatasetForAugment.distort_pts(hand_verts_r, data['distort_coeffs'])
                    hand_verts_l = ArcticDatasetForAugment.distort_pts(hand_verts_l, data['distort_coeffs'])
                    hand_joints_r = ArcticDatasetForAugment.distort_pts(hand_joints_r, data['distort_coeffs'])
                    hand_joints_l = ArcticDatasetForAugment.distort_pts(hand_joints_l, data['distort_coeffs'])
                    obj_verts = ArcticDatasetForAugment.distort_pts(obj_verts, data['distort_coeffs'])

                left_hand_faces  = manolayer['left'].faces[None].expand(num_random_per_finger + 1, -1, -1)
                right_hand_faces = manolayer['right'].faces[None].expand(num_random_per_finger + 1, -1, -1)
                scene = ObjectHandScene(_img_size_p1, _bbox_size,
                                        data['cameraK'], data['projectK'],
                                        obj_verts * 1000, obj_faces,
                                        hand_verts_l * 1000, left_hand_faces,  hand_joints_l * 1000,
                                        hand_verts_r * 1000, right_hand_faces, hand_joints_r * 1000,)
                checker = HandStatusCheck(p3d_renderer, scene)

                joint_vis, hand_mask, obj_mask = checker.check_joint_visibility(return_mask=True) # [N+1, 42], [N+1, H, W]
                joint_vis = joint_vis & checker.check_joint_outside_image().logical_not() # [N+1, 42], inside image
                joint_col = checker.check_joint_collision() # [N+1,], if sample has collision

                # 1) filter the sample with collision
                valid_ind = joint_col[1:].logical_not()

                # 2) joint visiblity should be same as the original sample
                original_occ_score = checker.determine_finger_occlusion_score(real_Jvis[data_idx].reshape(-1, 21)).reshape(-1, 5)
                random_occ_score   = checker.determine_finger_occlusion_score(joint_vis[1:].reshape(-1, 21)).reshape(-1, 5)
                original_occ_score = convert_fingerOrder(original_occ_score, 'nature2mano').reshape(-1, 10)
                random_occ_score   = convert_fingerOrder(random_occ_score, 'nature2mano').reshape(-1, 10)
                score_err = (random_occ_score[:, finger_idx] - original_occ_score[:, finger_idx]).abs()
                valid_ind = valid_ind & (score_err < 1e-3)

                # 3) visible 2d joint should be same as the original sample
                hand_joints_l_2d = batch_project_K(hand_joints_l, data['projectK'])[:, 1:] # [N+1, 20, 2]
                hand_joints_r_2d = batch_project_K(hand_joints_r, data['projectK'])[:, 1:] # [N+1, 20, 2]
                hand_joints_2d = torch.cat([hand_joints_l_2d, hand_joints_r_2d], dim=1) # [N+1, 40, 2]
                hand_joints_dist_2d = (hand_joints_2d[1:] - hand_joints_2d[:1]).square().sum(-1).sqrt() # [N, 40]
                hand_joints_dist_2d = hand_joints_dist_2d[:, real_finger_joint_vis.bool().reshape(-1)].mean(dim=-1) # [N,]
                valid_ind = valid_ind & (hand_joints_dist_2d < 0.5)

                # # 3) reprojected hand region should be the same as the original sample
                # reproj_iou = (hand_mask[:1] * hand_mask[1:]).sum(dim=(-1,-2)) \
                #            / ((1 - (1 - hand_mask[:1]) * (1 - hand_mask[1:])).sum(dim=(-1,-2)) + 1e-8)
                # reproj_valid = (reproj_iou > 0.97) | (hand_mask[:1].sum() < 100.) # high iou or invisible original hand region.
                # valid_ind = valid_ind & reproj_valid

                if valid_ind.any(): # cat the original sample
                    multi_finger_rvc = rotvec_randn[valid_ind, finger_idx] # [N, 3, 3]
                    multi_finger_rvc = torch.cat([original_finger_poses[:, finger_idx], multi_finger_rvc], dim=0)
                    multi_finger_poses[finger_idx] = multi_finger_rvc.cpu().numpy()
                else:
                    multi_finger_poses[finger_idx] = original_finger_poses[:, finger_idx].cpu().numpy()

            if not data['hand_valid_l']: # see invalid hand as invisible hand.
                multi_finger_poses[:5] = [rotvec_randn[:, finger_i].cpu().numpy() for finger_i in range(5)]
            if not data['hand_valid_r']:
                multi_finger_poses[5:] = [rotvec_randn[:, finger_i].cpu().numpy() for finger_i in range(5, 10)]


        file_name = f'{mode}_{protocal}/{data_idx//10000:03d}/multi_pose_{data_idx:06d}.pkl'
        save_path = osp.join(dataset.data_dir, 'MultiPoseAnnos', file_name)
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        pickle.dump(multi_finger_poses, open(save_path, 'wb'))





# -------------
def test_Arctic_multiAnno(mode:str, protocal:str, device:str=torch.device('cuda', 3)):

    p3d_renderer = Pytorch3DRenderer(image_size=_bbox_size, device=device)
    manolayer = {
        'right':RotVectorManolayer(model_path='checkpoints/processed_mano', hand_type='right',
                                   finger_tips_mode='smplx').to(device=device),
        'left': RotVectorManolayer(model_path='checkpoints/processed_mano', hand_type='left',
                                   finger_tips_mode='smplx', fix_left_shapedirs=False).to(device=device)}
    dataset = ArcticDatasetForAugment('/root/Workspace/DATASETS/ARCTIC',
                                      mode=mode, protocal=protocal, resolution=_bbox_size[0])
    real_Jvis = pickle.load(open(osp.join(dataset.data_dir, f'{mode}_{protocal}_visibility.pkl'), 'rb'))
    real_Jvis = torch.from_numpy(real_Jvis.reshape(-1, 2, 21)[..., 1:]).to(device=device) # [len_data, 2, 20], left hand and right fingers.

    # test_ind = pickle.load(open('test_ind.pkl', 'rb'))
    # for data_idx in test_ind:
    #     data_idx = test_ind[random.randint(0, len(test_ind)-1)]

    for data_idx in range(len(dataset)):

        data_idx = random.randint(0, len(dataset)) # fixme: for debug

        file_name = f'{mode}_{protocal}/{data_idx//10000:03d}/multi_pose_{data_idx:06d}.pkl'
        file_path = osp.join(dataset.data_dir, 'MultiPoseAnnos', file_name)
        multi_finger_pose = pickle.load(open(file_path, 'rb'))

        num_amb = 10
        amb_rotvec_l = np.zeros([num_amb, 5, 3, 3])
        amb_rotvec_r = np.zeros([num_amb, 5, 3, 3])

        for finger_idx in range(10):
            ambrvs = multi_finger_pose[finger_idx]
            random_ind = np.random.randint(0, len(ambrvs), num_amb)
            if finger_idx < 5:
                amb_rotvec_l[:, finger_idx] = ambrvs[random_ind]
            else:
                amb_rotvec_r[:, finger_idx-5] = ambrvs[random_ind]

        data = dataset[data_idx]
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key][None].to(device=device) # add batch dim

        root_rotvec_l = data['mano_rotvec_l'][:, :1].reshape(-1, 1, 3).repeat(num_amb, 1, 1)
        finger_rotvec_l = torch.from_numpy(amb_rotvec_l).float().to(device=device).reshape(-1, 15, 3)
        root_rotvec_r = data['mano_rotvec_r'][:, :1].reshape(-1, 1, 3).repeat(num_amb, 1, 1)
        finger_rotvec_r = torch.from_numpy(amb_rotvec_r).float().to(device=device).reshape(-1, 15, 3)
        shape_l, shape_r = data['mano_shape_l'].expand(num_amb, -1), data['mano_shape_r'].expand(num_amb, -1)
        trans_l, trans_r = data['mano_trans_l'].expand(num_amb, -1), data['mano_trans_r'].expand(num_amb, -1)

        hand_verts_l, hand_joints_l = manolayer['left' ](root_rotvec_l, finger_rotvec_l, shape_l, trans_l)
        hand_verts_r, hand_joints_r = manolayer['right'](root_rotvec_r, finger_rotvec_r, shape_r, trans_r)
        hand_faces_l = manolayer['left'].faces[None].expand(num_amb, -1, -1)
        hand_faces_r = manolayer['right'].faces[None].expand(num_amb, -1, -1)
        obj_verts, obj_faces = data['obj_verts'], data['obj_faces']

        if protocal == 'p2': # ego view should consider the camera distortion before projecting to the image plane.
            hand_verts_r = ArcticDatasetForAugment.distort_pts(hand_verts_r, data['distort_coeffs'])
            hand_verts_l = ArcticDatasetForAugment.distort_pts(hand_verts_l, data['distort_coeffs'])
            hand_joints_r = ArcticDatasetForAugment.distort_pts(hand_joints_r, data['distort_coeffs'])
            hand_joints_l = ArcticDatasetForAugment.distort_pts(hand_joints_l, data['distort_coeffs'])
            obj_verts = ArcticDatasetForAugment.distort_pts(obj_verts, data['distort_coeffs'])

        print(data['hand_valid_l'], data['hand_valid_r'])
        print(real_Jvis[data_idx].reshape(10, 4))

        brp = BrowserPlot(2, 1)
        brp.image(data['image'][0].cpu().numpy(), pos=[1,1], name=f'{data_idx}_image')
        brp.mesh3D(obj_verts[0].cpu().numpy(), obj_faces[0].cpu().numpy(), pos=[2,1], name=f'{data_idx}_object')
        for i in range(num_amb):
            brp.mesh3D(hand_verts_l[i].cpu().numpy(), hand_faces_l[i].cpu().numpy(), pos=[2,1], name=f'{data_idx}_hand_l_{i}')
            brp.mesh3D(hand_verts_r[i].cpu().numpy(), hand_faces_r[i].cpu().numpy(), pos=[2,1], name=f'{data_idx}_hand_r_{i}')
        brp.show(61234)


        exit()




if __name__ == '__main__':

    get_Arctic_visibility('val', 'p1')

    # # augment_Arctic_multiAnno('train', 'p2')
    # test_Arctic_multiAnno('train', 'p1')



