
import os
import sys

sys.path.append(os.getcwd())

import json
import os.path as osp
import pickle

import cv2
import numpy as np
import torch
from pytorch3d.io import load_obj
from tqdm import tqdm

from main.modules.Manolayer import (Mano_Rotvec_Mean_Left,
                                    Mano_Rotvec_Mean_Right, RotVectorManolayer)
from tools.browserPloty import BrowserPlot
from tools.dataTransform import (apply_crop_resize, compute_bbox_M, rvc_2_rmt,
                                 scale2d, transform2d)


# ------------------------------
def _padding(x:np.ndarray, pad_value:float=1.0):

    N,_ = x.shape
    x_pad = np.ones((N, 1), dtype=np.float32) * pad_value

    return np.concatenate([x, x_pad], axis=-1)

# ------------------------------
def convert_ARCTIC_bbox(cx, cy, scale): # [cx, cy, scale] -> [x1, y1, len_x, len_y]

    edge_length = 300 * scale
    x1 = cx - edge_length / 2
    y1 = cy - edge_length / 2

    return [x1, y1, edge_length, edge_length]

# ------------------------------
def convert_ARCTIC_hand_joint_order(joints):
    """
    Convert the order of joints from MANO to Natural order.
    """
    joints = joints[[0, 13, 14, 15, 16,
                         1,  2,  3, 17,
                         4,  5,  6, 18,
                        10, 11, 12, 19,
                         7,  8,  9, 20]] # to nature order
    return joints

# ------------------------------
Manolayers = {
    'left':  RotVectorManolayer(model_path='checkpoints/processed_mano', hand_type='left',
                                fix_left_shapedirs=False, set_root_as_origin=True, finger_tips_mode='smplx'),
    'right': RotVectorManolayer(model_path='checkpoints/processed_mano', hand_type='right',
                                set_root_as_origin=True, finger_tips_mode='smplx')
}
ManoRvcs = {
    'left':  np.array(Mano_Rotvec_Mean_Left),
    'right': np.array(Mano_Rotvec_Mean_Right),
}
ObjectNames = [
    "capsulemachine",
    "box",
    "ketchup",
    "laptop",
    "microwave",
    "mixer",
    "notebook",
    "espressomachine",
    "waffleiron",
    "scissors",
    "phone",]


## =========================
class OriginalArcticReader:
    """
    world space -> camera space -> cropped camera space.
    Offical ARCTIC dataset reader providing the cropped camera space data.
    Some operation is refer to the official implements.
    """
    def __init__(self,
                 data_dir:str='/root/Workspace/DATASETS/ARCTIC',
                 mode:str='val',
                 protocal:str='p1',
                 **kwargs) -> None:

        original_data = np.load(osp.join(data_dir, 'splits', f'{protocal}_{mode}.npy'),
                                allow_pickle=True).item()
        self.mode = mode
        self.protocal = protocal

        self.image_dir = osp.join(data_dir, 'cropped_images')

        self.data_dicts = original_data['data_dict']
        self.image_names = original_data['imgnames']
        self.meta_misc = json.load(open(osp.join(data_dir, 'meta', 'misc.json'), 'r'))

        self.objects = pickle.load(open(osp.join(data_dir, 'custom_annotations/simplified_objects/faces_max_10K.pkl'), 'rb'))

    # ------------------------------
    def __len__(self):
        return len(self.image_names)

    # ------------------------------
    def __getitem__(self, idx):

        # 1. split the image name to get the necessary information.
        subject_id, sequence_name, view_idx, basename = self.image_names[idx].split('/')[-4:] # str
        img_imread_name = osp.join(self.image_dir, f'{subject_id}/{sequence_name}/{view_idx}/{basename}')
        frame_idx, view_idx = int(basename.split('.')[0]) - self.meta_misc[subject_id]["ioi_offset"], int(view_idx)
        obj_name = sequence_name.split('_')[0]
        is_ego = True if view_idx == 0 else False # ego view has different camera parameters.
        index_chain = f'{subject_id}.{sequence_name}.{view_idx:1d}.{frame_idx:05d}'

        # 2. load the image and get the necessary data.
        img = None # cv2.cvtColor(cv2.imread(img_imread_name), cv2.COLOR_BGR2RGB) # fixme:
        seq_data = self.data_dicts[f'{subject_id}/{sequence_name}']

        # 3. process the transformation of object
        obj_arti_rvc  = np.array([0., 0., -1 * seq_data['params']['obj_arti'][frame_idx]]) # articulation rot around z-axis.
        obj_world_rvc = seq_data['params']['obj_rot'][frame_idx]
        obj_world_trans = seq_data['params']['obj_trans'][frame_idx] / 1000.0 # mm -> m

        obj_arti_rmt  = rvc_2_rmt(obj_arti_rvc.reshape(1, 3)).reshape(3, 3)
        obj_world_rmt = rvc_2_rmt(obj_world_rvc.reshape(1, 3)).reshape(3, 3)

        if is_ego:
            R_world2cam = seq_data['params']['world2ego'][frame_idx]
        else:
            R_world2cam = np.array(self.meta_misc[subject_id]['world2cam'])[view_idx-1]

        # 4. aggregate the transformation of object.
        top_world2cam_Rt = np.eye(4) # top of the object will be applied a extra articulation rotation.
        top_world2cam_Rt[:3, :3] = obj_world_rmt @ obj_arti_rmt
        top_world2cam_Rt[:3,  3] = obj_world_trans
        top_world2cam_Rt = R_world2cam @ top_world2cam_Rt

        bottom_world2cam_Rt = np.eye(4)
        bottom_world2cam_Rt[:3, :3] = obj_world_rmt
        bottom_world2cam_Rt[:3,  3] = obj_world_trans
        bottom_world2cam_Rt = R_world2cam @ bottom_world2cam_Rt

        # 5. process the transformation of hand.
        # img_size = img.shape[:2]
        img_size = (600, 840) if is_ego else (1000, 1000) # H, W
        bbox = convert_ARCTIC_bbox(*seq_data['bbox'][frame_idx, view_idx])
        bbox_M = compute_bbox_M(bbox, dst_size=img_size[::-1]) if not is_ego else scale2d(0.3, 0.3)

        if is_ego:
            cameraK = seq_data['params']['K_ego'][frame_idx]
            distort_coeffs = seq_data['params']['dist'][frame_idx]
        else:
            cameraK = np.array(self.meta_misc[subject_id]["intris_mat"][view_idx - 1])
            distort_coeffs = None
        cameraK = bbox_M @ cameraK

        j2d_l = transform2d(seq_data['2d']['joints.left' ][frame_idx, view_idx], bbox_M)
        j2d_r = transform2d(seq_data['2d']['joints.right'][frame_idx, view_idx], bbox_M)
        j2d_l = convert_ARCTIC_hand_joint_order(j2d_l) # to nature order
        j2d_r = convert_ARCTIC_hand_joint_order(j2d_r)

        j3d_l = seq_data['cam_coord']['joints.left'][frame_idx, view_idx]
        j3d_r = seq_data['cam_coord']['joints.right'][frame_idx, view_idx]
        j3d_l = convert_ARCTIC_hand_joint_order(j3d_l)
        j3d_r = convert_ARCTIC_hand_joint_order(j3d_r)

        # 6. process the hand mano.
        mano_pose_l  = np.concatenate([seq_data['cam_coord']['rot_l_cam'][frame_idx, view_idx],
                                       seq_data['params']['pose_l'][frame_idx] + ManoRvcs['left'][3:]], axis=0)
        mano_shape_l = seq_data['params']['shape_l'][frame_idx]

        mano_pose_r  = np.concatenate([seq_data['cam_coord']['rot_r_cam'][frame_idx, view_idx],
                                       seq_data['params']['pose_r'][frame_idx] + ManoRvcs['right'][3:]], axis=0)
        mano_shape_r = seq_data['params']['shape_r'][frame_idx]

        hand_valid_l = seq_data['cam_coord']['is_valid'][frame_idx, view_idx] * \
            seq_data['cam_coord']['left_valid'][frame_idx, view_idx] # [1,]
        hand_valid_r = seq_data['cam_coord']['is_valid'][frame_idx, view_idx] * \
            seq_data['cam_coord']['right_valid'][frame_idx, view_idx]

        return {
            'index_chain': index_chain,
            'img': img,  'image_size': img_size,
            'cameraK': cameraK, 'distort_coeffs': distort_coeffs,

            'j2d_l': j2d_l, 'j2d_r': j2d_r,
            'j3d_l': j3d_l, 'j3d_r': j3d_r,

            'mano_pose_l': mano_pose_l, 'mano_shape_l': mano_shape_l,
            'mano_pose_r': mano_pose_r, 'mano_shape_r': mano_shape_r,

            'obj_name': obj_name,
            'obj_top_world2cam_Rt': top_world2cam_Rt,
            'obj_bottom_world2cam_Rt': bottom_world2cam_Rt,

            'hand_valid_l': hand_valid_l, # NOTE: for evaluation, same as the official implement of ARCTIC.
            'hand_valid_r': hand_valid_r,
        }

    # ------------------------------
    def transform_artic_object(self, obj_name:str,
                               top_world2cam_Rt:np.ndarray, bottom_world2cam_Rt:np.ndarray): # 4 x 4 array

        obj_meshes = self.objects[obj_name]
        obj_verts_top = (_padding(obj_meshes['verts_top']) @ top_world2cam_Rt.T)[:, :3]
        obj_verts_bottom = (_padding(obj_meshes['verts_bottom']) @ bottom_world2cam_Rt.T)[:, :3]

        return obj_verts_top, obj_verts_bottom

    # ------------------------------
    @staticmethod
    def distort_pts(pts:np.ndarray, coeffs:np.ndarray): # NOTE distort the 3D points in ego camera space before projection.

        assert pts.ndim == 2 and pts.shape[1] == 3, 'pts should be Nx3-dim.'
        assert len(coeffs) == 8, 'coefficients should be 8-dim.'
        eps = 1e-8

        z = pts[..., 2]
        x1, y1 = pts[..., 0] / (z + eps), pts[..., 1] / (z + eps)
        x1_2, y1_2, x1_y1 = x1 * x1, y1 * y1, x1 * y1

        r2 = x1_2 + y1_2
        r4 = r2 * r2
        r6 = r4 * r2

        r_dist = (1 + coeffs[0] * r2 + coeffs[1] * r4 + coeffs[4] * r6) \
               / (1 + coeffs[5] * r2 + coeffs[6] * r4 + coeffs[7] * r6)

        # full (rational + tangential) distortion
        x2 = x1 * r_dist + 2 * coeffs[2] * x1_y1 + coeffs[3] * (r2 + 2 * x1_2)
        y2 = y1 * r_dist + 2 * coeffs[3] * x1_y1 + coeffs[2] * (r2 + 2 * y1_2)

        # denormalize for projection (which is a linear operation)
        distored_pts = np.stack([x2 * z, y2 * z, z], axis=-1)

        return distored_pts



# ------------------------------
def arange_object_data(data_dir:str='/root/Workspace/DATASETS/ARCTIC'):

    obj_template_dir = osp.join(data_dir, 'meta/object_vtemplates')
    object_names = sorted(os.listdir(obj_template_dir))

    obj_data = dict()
    for obj_name in object_names:
        mesh_path = osp.join(obj_template_dir, obj_name, 'mesh.obj')
        mesh_data = load_obj(mesh_path)
        obj_verts, obj_faces = mesh_data[0].numpy() / 1000, mesh_data[1].verts_idx.numpy()

        parts_info = json.load(open(osp.join(obj_template_dir, obj_name, 'parts.json'), 'r'))
        is_top = (np.array(parts_info) == 0)
        # is_bottom = ~is_top

        obj_data[obj_name] = dict(verts=obj_verts,
                                  faces=obj_faces,
                                  is_top=is_top)

    save_name = osp.join(data_dir, 'custom_annotations', 'official_simplified_objects.pkl')
    pickle.dump(obj_data, open(save_name, 'wb'))




# ------------------------------
def arange_custom_annotation(data_dir:str='/root/Workspace/DATASETS/ARCTIC',
                             mode:str='train', protocal:str='p1'):
    """
    Rearrange the custom annotation data.
    """
    # NOTE: The following images not exists, relaced by the previous one in sqeuence.
    # "s08/mixer_use_02/3/00295.jpg" <- "s08/mixer_use_02/3/00294.jpg"
    # "s01/laptop_use_04/7/00737.jpg" <- "s01/laptop_use_04/7/00736.jpg"
    # "s01/espressomachine_grab_01/4/00443.jpg" <- "s01/espressomachine_grab_01/4/00442.jpg"
    data_reader = OriginalArcticReader(data_dir, mode, protocal)

    index_chain_list = []
    for idx in tqdm(range(len(data_reader))):

        # import random
        # idx = random.randint(0, len(data_reader)-1)

        data = data_reader[idx]
        index_chain_list.append(data['index_chain'])

        image_anno = dict()

        # 1. base information
        image_anno['cameraK'] = data['cameraK'].reshape(3, 3).astype(np.float32)
        image_anno['distort_coeffs'] = data['distort_coeffs'].reshape(-1).astype(np.float32) \
            if data['distort_coeffs'] is not None else None

        # 2. mano annotation
        pose_r_tensor = torch.from_numpy(data['mano_pose_r']).reshape(16, 3).unsqueeze(0)
        shape_r_tensor = torch.from_numpy(data['mano_shape_r']).reshape(10).unsqueeze(0)
        v3d_r, j3d_r = Manolayers['right'](pose_r_tensor[:, :1], pose_r_tensor[:, 1:], shape_r_tensor)
        v3d_r, j3d_r = v3d_r.squeeze(0).detach().numpy(), j3d_r.squeeze(0).detach().numpy()
        trans_r = (data['j3d_r'] - j3d_r).mean(axis=0)

        pose_l_tensor = torch.from_numpy(data['mano_pose_l']).reshape(16, 3).unsqueeze(0)
        shape_l_tensor = torch.from_numpy(data['mano_shape_l']).reshape(10).unsqueeze(0)
        v3d_l, j3d_l = Manolayers['left'](pose_l_tensor[:, :1], pose_l_tensor[:, 1:], shape_l_tensor)
        v3d_l, j3d_l = v3d_l.squeeze(0).detach().numpy(), j3d_l.squeeze(0).detach().numpy()
        trans_l = (data['j3d_l'] - j3d_l).mean(axis=0) # [3,]

        image_anno['mano_pose_r'] = data['mano_pose_r'].reshape(16, 3).astype(np.float32)
        image_anno['mano_pose_l'] = data['mano_pose_l'].reshape(16, 3).astype(np.float32)
        image_anno['mano_shape_r'] = data['mano_shape_r'].reshape(10).astype(np.float32)
        image_anno['mano_shape_l'] = data['mano_shape_l'].reshape(10).astype(np.float32)
        image_anno['mano_trans_r'] = trans_r.reshape(3).astype(np.float32)
        image_anno['mano_trans_l'] = trans_l.reshape(3).astype(np.float32)

        # 3. joint annotation
        H, W = data['image_size']
        j2d_l, j2d_r = data['j2d_l'], data['j2d_r']
        valid_l = (j2d_l[:, 0] >= 0) & (j2d_l[:, 0] <= W) & (j2d_l[:, 1] >= 0) & (j2d_l[:, 1] <= H)
        valid_r = (j2d_r[:, 0] >= 0) & (j2d_r[:, 0] <= W) & (j2d_r[:, 1] >= 0) & (j2d_r[:, 1] <= H)
        valid = (valid_l.any() or valid_r.any())

        image_anno['j2d_r'] = data['j2d_r'].reshape(21, 2).astype(np.float32)
        image_anno['j2d_l'] = data['j2d_l'].reshape(21, 2).astype(np.float32)
        image_anno['j3d_r'] = data['j3d_r'].reshape(21, 3).astype(np.float32)
        image_anno['j3d_l'] = data['j3d_l'].reshape(21, 3).astype(np.float32)
        image_anno['valid_r'] = valid_r.astype(np.bool_) # [21,], custom valid informations.
        image_anno['valid_l'] = valid_l.astype(np.bool_) # [21,]
        image_anno['valid_both'] = valid.astype(np.bool_) # [1,]
        image_anno['hand_valid_r'] = data['hand_valid_r'].astype(np.bool_) # [1,] official valid informations for hand.
        image_anno['hand_valid_l'] = data['hand_valid_l'].astype(np.bool_) # [1,]

        # 4. object annotation
        image_anno['obj_name'] = data['obj_name']
        image_anno['obj_top_world2cam_Rt'] = data['obj_top_world2cam_Rt'].reshape(4, 4).astype(np.float32)
        image_anno['obj_bottom_world2cam_Rt'] = data['obj_bottom_world2cam_Rt'].reshape(4, 4).astype(np.float32)

        # 5. save the annotation
        base_name = data['index_chain'].replace('.', '/') + '.pkl'
        save_name = osp.join(data_dir, 'custom_annotations', base_name)
        os.makedirs(osp.dirname(save_name), exist_ok=True)
        pickle.dump(image_anno, open(save_name, 'wb'))

    pickle.dump(index_chain_list, open(osp.join(data_dir, 'custom_annotations',
                                                f'{mode}_{protocal}_index_chain.pkl'), 'wb'))





# ------------------------------
if __name__ == '__main__':

    # # ================================================================================
    # 1. generate the simplified object data. (preprocessed in Blender)
    # # ================================================================================

    # import pickle
    # import time

    # obj_dir = '/root/Workspace/DATASETS/ARCTIC/custom_annotations/simplified_objects/faces_max_10000'
    # obj_pkl = dict()
    # for obj_name in ObjectNames:

    #     obj_mesh = load_obj(osp.join(obj_dir, f'{obj_name}.obj'))
    #     obj_verts, obj_faces = obj_mesh[0].numpy() / 1000, obj_mesh[1].verts_idx.numpy()

    #     obj_mesh_top = load_obj(osp.join(obj_dir, f'{obj_name}_top.obj'))
    #     obj_verts_top, obj_faces_top = obj_mesh_top[0].numpy() / 1000, obj_mesh_top[1].verts_idx.numpy()

    #     obj_mesh_bottom = load_obj(osp.join(obj_dir, f'{obj_name}_bottom.obj'))
    #     obj_verts_bottom, obj_faces_bottom = obj_mesh_bottom[0].numpy() / 1000, obj_mesh_bottom[1].verts_idx.numpy()

    #     obj_pkl[obj_name] = {
    #         'verts': obj_verts.astype(np.float32), 'faces': obj_faces.astype(np.int32),
    #         'verts_top': obj_verts_top.astype(np.float32), 'faces_top': obj_faces_top.astype(np.int32),
    #         'verts_bottom': obj_verts_bottom.astype(np.float32), 'faces_bottom': obj_faces_bottom.astype(np.int32),
    #     }
    #     # brp = BrowserPlot(3, 1)
    #     # brp.mesh3D(obj_verts, obj_faces, pos=[1,1])
    #     # brp.mesh3D(obj_verts_top, obj_faces_top, pos=[2,1])
    #     # brp.mesh3D(obj_verts_bottom, obj_faces_bottom, pos=[3,1])
    #     # brp.show(61234)
    #     # exit()

    # pickle.dump(obj_pkl, open('/root/Workspace/DATASETS/ARCTIC/custom_annotations/simplified_objects/faces_max_10K.pkl', 'wb'))
    # exit()


    # # ================================================================================
    # # 2. test the data reader.
    # # ================================================================================

    # reader = OriginalArcticReader(mode='val', protocal='p2')
    # print('len:', len(reader))

    # # counter = 0
    # # for idx,name in enumerate(reader.image_names):
    # #     subject_id, sequence_name, view_idx, basename = name.split('/')[-4:] # str
    # #     if view_idx == 0 or view_idx == '0':
    # #         # print(idx)
    # #         counter += 1
    # # print('counter:', counter)
    # # exit()

    # for idx in range(50):

    #     idx = np.random.randint(0, len(reader)-1)
    #     data = reader[idx]

    #     # print(idx, data['image_size'])
    #     # continue

    #     pose_r_tensor = torch.from_numpy(data['mano_pose_r']).reshape(16, 3).unsqueeze(0)
    #     shape_r_tensor = torch.from_numpy(data['mano_shape_r']).reshape(10).unsqueeze(0)
    #     v3d_r, j3d_r = Manolayers['right'](pose_r_tensor[:, :1], pose_r_tensor[:, 1:], shape_r_tensor)
    #     v3d_r, j3d_r = v3d_r.squeeze(0).detach().numpy(), j3d_r.squeeze(0).detach().numpy()
    #     trans_r = (data['j3d_r'] - j3d_r).mean(axis=0)
    #     v3d_r += trans_r

    #     pose_l_tensor = torch.from_numpy(data['mano_pose_l']).reshape(16, 3).unsqueeze(0)
    #     shape_l_tensor = torch.from_numpy(data['mano_shape_l']).reshape(10).unsqueeze(0)
    #     v3d_l, j3d_l = Manolayers['left'](pose_l_tensor[:, :1], pose_l_tensor[:, 1:], shape_l_tensor)
    #     v3d_l, j3d_l = v3d_l.squeeze(0).detach().numpy(), j3d_l.squeeze(0).detach().numpy()
    #     trans_l = (data['j3d_l'] - j3d_l).mean(axis=0)
    #     v3d_l += trans_l

    #     # print(j3d_r - data['j3d_r'])
    #     # print(j3d_l - data['j3d_l'])

    #     obj_verts_top, obj_verts_bottom = reader.transform_artic_object(
    #         data['obj_name'], data['obj_top_world2cam_Rt'], data['obj_bottom_world2cam_Rt'])
    #     obj_faces_top = reader.objects[data['obj_name']]['faces_top']
    #     obj_faces_bottom = reader.objects[data['obj_name']]['faces_bottom']

    #     if data['distort_coeffs'] is not None:
    #         v3d_r_x = reader.distort_pts(v3d_r, data['distort_coeffs'])
    #         v3d_l = reader.distort_pts(v3d_l, data['distort_coeffs'])
    #         obj_verts_top = reader.distort_pts(obj_verts_top, data['distort_coeffs'])
    #         obj_verts_bottom = reader.distort_pts(obj_verts_bottom, data['distort_coeffs'])

    #     obj_verts_proj = np.concatenate([(obj_verts_top / obj_verts_top[:, 2:]) @ data['cameraK'].T,
    #                                      (obj_verts_bottom / obj_verts_bottom[:, 2:]) @ data['cameraK'].T],
    #                                      axis=0)[:, :2]
    #     hand_verts_proj = (np.concatenate([v3d_l / (v3d_l[:, 2:] + 1e-5),
    #                                        v3d_r / (v3d_r[:, 2:] + 1e-5)], axis=0) @ data['cameraK'].T)[:, :2]
    #     hand_joints_proj = (np.concatenate([data['j3d_l'] / (data['j3d_l'][:, 2:] + 1e-5),
    #                                         data['j3d_r'] / (data['j3d_r'][:, 2:] + 1e-5)], axis=0) @ data['cameraK'].T)[:, :2]

    #     print('Left  Joints Reproject Err: ', np.linalg.norm(hand_joints_proj[:21] - data['j2d_l'], 2, -1).mean())
    #     print('Right Joints Reproject Err: ', np.linalg.norm(hand_joints_proj[21:] - data['j2d_r'], 2, -1).mean())

    #     brp = BrowserPlot(2, 1)

    #     brp.image(data['img'], pos=[1,1], name=f'{idx}')
    #     brp.scatter2D(obj_verts_proj, color='red', pos=[1,1])
    #     brp.scatter2D(hand_verts_proj, color='blue', pos=[1,1])
    #     brp.scatter2D(data['j2d_l'], color='green', pos=[1,1])
    #     brp.scatter2D(data['j2d_r'], color='yellow', pos=[1,1])

    #     brp.mesh3D(obj_verts_top, obj_faces_top, pos=[2,1])
    #     brp.mesh3D(obj_verts_bottom, obj_faces_bottom, pos=[2,1])
    #     brp.mesh3D(v3d_r, Manolayers['right'].faces.numpy(), pos=[2,1])
    #     brp.mesh3D(v3d_l, Manolayers['left'].faces.numpy(), pos=[2,1])
    #     brp.show(61234)
    #     # brp.save_html(f'_rundir/test/{reader.mode}_{reader.protocal}_{idx}.html')



    # # ================================================================================
    # # 4. produce the custom object data from official simplified object templates.
    # # ================================================================================
    # arange_object_data()


    # # ================================================================================
    # # 4. produce the custom annotation
    # # ================================================================================
    for mode in ['val','train']:
        for protocal in ['p1', 'p2']:
            arange_custom_annotation(mode=mode, protocal=protocal)