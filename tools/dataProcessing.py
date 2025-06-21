
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from numpy import ndarray


## ---------------------------
def get_procrustes_alignment_ndarray(
        src:np.ndarray, tgt:np.ndarray, exclude_R:bool=False) -> Tuple:
    """Do Procrustes Alignment

    Parameter:
    ----
    src: ndarray
        [B, N, P], source feature array with B batch and P feature vectors, P always 2 or 3 when input verts or joints
    tgt: ndarray
        [B, N, P], tgt feature array.
    exclude_R: bool
        if exclude Rotation when compute align.

    Returns:
    ----
    src_aligned : ndarray
        [B, N, P], Aligned src array.
    rst: 3 x ndarray
        Rotation-[B, P, P], Scale-[B], T-[B, P], transformations for aligning src to tgt with formula: src @ r * s + t.
    """
    assert src.shape == tgt.shape
    muX = np.mean(tgt, axis=1, keepdims=True)
    muY = np.mean(src, axis=1, keepdims=True)

    X0 = tgt - muX
    Y0 = src - muY
    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True) + 1e-8)
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True) + 1e-8)

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
    if not exclude_R:
        t = muX - a * np.matmul(muY, R) # Translation
        src_aligned = a * np.matmul(src, R) + t
        return src_aligned, (R, a[:,0], t[:,0])
    else:
        t = muX - a * muY
        src_aligned = a * src + t
        return src_aligned, (None, a[:,0], t[:,0])



# --------------------------
def get_procrustes_alignment_tensor(
        src:torch.Tensor, tgt:torch.Tensor, exclude_R:bool=False) -> Tuple:
    """ See the method "get_procrustes_alignment_ndarray".
    """
    assert src.shape == tgt.shape
    muX = torch.mean(tgt, dim=1, keepdim=True)
    muY = torch.mean(src, dim=1, keepdim=True)

    X0 = tgt - muX
    Y0 = src - muY
    normX = torch.sqrt(torch.sum(X0**2, dim=(1, 2), keepdim=True) + 1e-8)
    normY = torch.sqrt(torch.sum(Y0**2, dim=(1, 2), keepdim=True) + 1e-8)

    X0 /= normX
    Y0 /= normY

    H = torch.matmul(X0.transpose(1, 2), Y0)
    U, s, Vt = torch.linalg.svd(H)
    V = Vt.transpose(1, 2)
    R = torch.matmul(V, U.transpose(1, 2))

    sign_detR = torch.sign(torch.linalg.det(R).unsqueeze(1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = torch.matmul(V, U.transpose(1, 2))

    tr = torch.sum(s, dim=1, keepdim=True).unsqueeze(2)

    a = tr * normX / normY
    if not exclude_R:
        t = muX - a * torch.matmul(muY, R)
        src_aligned = a * torch.matmul(src, R) + t
        return src_aligned, (R, a.squeeze(1), t.squeeze(1))
    else:
        t = muX - a * muY
        src_aligned = a * src + t
        return src_aligned, (None, a.squeeze(1), t.squeeze(1))




# # -----------------------
# def produce_joint2d_bbox(joint2d:ndarray, expansion:float=1.0):

#     xs, ys = joint2d[:,0], joint2d[:,1]
#     xmin, ymin = min(xs), min(ys)
#     xmax, ymax = max(xs), max(ys)

#     x_center = (xmin + xmax) / 2.
#     width = (xmax - xmin) * expansion
#     y_center = (ymin + ymax) / 2.
#     height = (ymax - ymin) * expansion

#     edge = max(width, height)
#     xmin = x_center - 0.5 * edge
#     ymin = y_center - 0.5 * edge

#     bbox = np.array([xmin, ymin, edge, edge]).astype(np.float32)

#     return bbox


# ## ------------
# def crop_image_via_bbox(src_img:ndarray, bbox:ndarray, dst_size:Tuple=(256,256)
#                         ) -> Tuple[ndarray, ndarray, ndarray]:
#     """
#     Crop and resize the image part inside bbox.

#     Parameters
#     ----
#     src_img : ndarray
#         `[H, W, C]`, uint8.
#     bbox : ndarray
#         `[Topleft-X, Topleft-Y, Width, Height]`, float.
#     dst_size : Tuple
#         `[width, height]`, the size of output affined image.

#     Returns
#     ----
#     dst_img : ndarray
#         `[H, W, C]`
#     M : ndarray
#         `[3, 3]`, transform matrix for crop and resize forward pass.
#     inv_M : ndarray
#         `[3, 3]`, inverse matrix for crop and resize backword pass.

#     """

#     src_topleft = np.array([bbox[0], bbox[1]], dtype=np.float32)
#     src_topright = np.array([bbox[0] + bbox[2], bbox[1]], dtype=np.float32)
#     src_bottomleft = np.array([bbox[0], bbox[1] + bbox[3]], dtype=np.float32)
#     src_group = np.stack([src_topleft, src_topright, src_bottomleft], axis=0)

#     dst_topleft = np.array([0., 0.], dtype=np.float32)
#     dst_topright = np.array([dst_size[0], 0.], dtype=np.float32)
#     dst_bottomleft = np.array([0., dst_size[1]], dtype=np.float32)
#     dst_group = np.stack([dst_topleft, dst_topright, dst_bottomleft], axis=0)

#     M = cv2.getAffineTransform(src_group, dst_group)
#     inv_M = cv2.getAffineTransform(dst_group, src_group)
#     dst_img = cv2.warpAffine(src_img, M, dsize=dst_size, flags=cv2.INTER_LINEAR) # border with 0.

#     affine_M = np.eye(3, dtype=np.float32)
#     affine_M[:2] = M

#     affine_M_inv = np.eye(3, dtype=np.float32)
#     affine_M_inv[:2] = inv_M

#     return dst_img, affine_M, affine_M_inv


# # -----------------
# def compute_bbox_M(bbox:Union[ndarray, List], # [topleft_x, topleft_y, len_x, len_y]
#                    dst_size:List # [W, H]
#                    ):

#     translate = translate2d(-bbox[0], -bbox[1])
#     scale = scale2d(dst_size[0] / bbox[2], dst_size[1] / bbox[3])
#     bbox_M = scale @ translate

#     return bbox_M



# # -----------------
# def apply_crop_resize(image:ndarray,
#                         bbox:Union[ndarray, List], # [topleft_x, topleft_y, len_x, len_y]
#                         dst_size:List, # [W, H]
#                         ) -> ndarray:
#     """ Apply crop and resize transformation on uint8 ndarray image.

#     Parameters
#     ----
#     image: ndarray
#         `[H, W, C]`, uint8 rgb.
#     bbox: ndarray, List
#         `[topleft_x, topleft_y, len_x, len_y]`
#     dst_size: List
#         `[W, H]`, target size.

#     Returns:
#     ----
#     crop_image: ndarray
#         `[h,w,c]`
#     crop_M: ndarray
#         the crop and resize transformation matrix.
#     """

#     translate = translate2d(-bbox[0], -bbox[1])
#     scale = scale2d(dst_size[0] / bbox[2], dst_size[1] / bbox[3])
#     crop_M = scale @ translate

#     image = cv2.warpAffine(image, crop_M[:2, :],
#                             dsize=dst_size,
#                             flags=cv2.INTER_LINEAR)
#     return image, crop_M



# ## ------------
# def transform2d(xy:ndarray, matrix:ndarray):
#     """
#     Applied 2d transformation on 2d points.
#     matrix: [2, 3] or [3, 3].
#     """
#     x = xy[:,0] * matrix[0,0] + xy[:,1] * matrix[0,1] + matrix[0,2]
#     y = xy[:,0] * matrix[1,0] + xy[:,1] * matrix[1,1] + matrix[1,2]

#     return np.stack([x, y], axis=-1)




# ## ------------
# def rotate2d(rad:float): # rotation angles in radians
#     cos, sin = np.cos(rad), np.sin(rad)
#     return np.array([[cos, -sin, 0],
#                      [sin,  cos, 0],
#                      [  0,    0, 1],], dtype=np.float32)

# ## ------------
# def translate2d(tx:float, ty:float):
#     return np.array([[1., 0., tx],
#                      [0., 1., ty],
#                      [0., 0., 1.],], dtype=np.float32)

# ## ------------
# def scale2d(sx:float, sy:float):
#     return np.array([[sx, 0., 0.],
#                      [0., sy, 0.],
#                      [0., 0., 1.],], dtype=np.float32)

# ## ------------
# def flip2d(H:int=None, W:int=None):  # if flip on H (Y axis), input H, if flip on H,W together, input H,W together
#     M = np.eye(3)
#     if W is not None:
#         M[0, 0], M[0, 2] = -1, W
#     if H is not None:
#         M[1, 1], M[1, 2] = -1, H
#     return M




# ## ===================
# class AugmentPipe:
#     def __init__(self,
#                  color_prob:float=1.0,
#                  noise_prob:float=1.0,
#                  geometry_prob:float=1.0) -> None:

#         self.color_prob = color_prob
#         self.noise_prob = noise_prob
#         self.geometry_prob = geometry_prob


#     # --------------------
#     def forward(self, image:ndarray, j2d:ndarray=None, j3d:ndarray=None, rootrot:ndarray=None, **kwargs):
#         """ Augment the data randomly.

#         Parameters
#         ----
#         image: ndarray
#             `[H, W, C]`, uint8 rgb.
#         j2d: ndarray
#             `[N, 2]`
#         j3d: ndarray
#             `[N, 3]`
#         rootrot: ndarray
#             `[3, 3]`, global rotation.

#         """

#         do_color_aug = np.random.rand(4) < self.color_prob
#         do_geome_aug = np.random.rand(3) < self.geometry_prob
#         do_noise_aug = np.random.rand(1) < self.noise_prob

#         color_strengths = [0.05, 0.25, 0.5, 0.2]
#         color_aug_values = np.random.normal(0.0, 0.4, 4).astype(np.float32).clip(-1., 1.)
#         hue = do_color_aug[0] * color_aug_values[0] * color_strengths[0]
#         sat = 1.0 + do_color_aug[1] * color_aug_values[1] * color_strengths[1]
#         bri = 1.0 + do_color_aug[2] * color_aug_values[2] * color_strengths[2]
#         con = 1.0 + do_color_aug[3] * color_aug_values[3] * color_strengths[3]

#         geome_strengths = [0.15, 1.0, 0.125]
#         geome_aug_values = np.random.normal(0.0, 0.4, 4).astype(np.float32).clip(-1., 1.)
#         scale   = 1.0 - (do_geome_aug[0] * geome_aug_values[0] * geome_strengths[0])
#         rotate  = do_geome_aug[1] * geome_aug_values[1] * geome_strengths[1]
#         trans_x = do_geome_aug[2] * geome_aug_values[2] * geome_strengths[2]
#         trans_y = do_geome_aug[2] * geome_aug_values[3] * geome_strengths[2]

#         image = self.apply_color_transform(image, hue, sat, bri, con) if self.color_prob != 0 else image
#         image, (S, R, T, M) = self.apply_geometry_transform(image, scale, rotate, [trans_x, trans_y])
#         image = self.add_guassian_noise(image, noise_strength=10.0) if do_noise_aug else image

#         j2d = transform2d(j2d, M) if j2d is not None else None
#         j3d = j3d @ R.T if j3d is not None else None
#         rootrot = R @ rootrot if rootrot is not None else None

#         return image, (j2d, j3d, rootrot)



#     # --------------------
#     @staticmethod
#     def apply_geometry_transform(image:np.ndarray,
#                                  scale:float=1.0,
#                                  rotation:float=0.0, # clockwise, in radiance
#                                  translation:Union[ndarray, List]=[0.0, 0.0], # [H, W]
#                                  ) -> ndarray:
#         """Apply geometric transformation with the image center as the origin.

#         Parameters
#         ----
#         image: ndarray
#             `[H, W, C]`, uint8 rgb.
#         scale: float
#             scale of image.
#         rotation: float
#             rotation of image. in radiance and clockwise
#         translation: list
#             translation of image. [H, W], ratio of image resolution. +X right, +Y down.

#         Returns:
#         ----
#         affine_M: ndarray
#             the integrated transformation matrix.

#         """
#         H, W, _ = image.shape

#         origin = [W/2, H/2]
#         trans_0 = [-origin[0], -origin[1]]
#         trans_1 = [origin[0], origin[1]]

#         S = scale2d(scale, scale)
#         R = rotate2d(rotation)
#         T = translate2d(translation[1]*W, translation[0]*H)
#         M = translate2d(*trans_1) @ T @ R @ S @ translate2d(*trans_0) # stay the center. scale -> rot -> trans

#         image = cv2.warpAffine(image, M[:2, :],
#                                dsize=[W, H],
#                                flags=cv2.INTER_LINEAR,)
#                             #    borderMode=cv2.BORDER_CONSTANT,
#                             #    borderValue=[124, 116, 103]) # the mean of imagenet dataset.

#         return image, (S, R, T, M)



#     # ---------------------
#     @staticmethod
#     def apply_color_transform(image:np.ndarray,
#                               hue:float=0.0,
#                               saturation:float=1.0,
#                               brightness:float=1.0,
#                               constrast:float=1.0,
#                               ) -> ndarray:
#         """Apply color transformation on uint8 ndarray image.

#         Parameters
#         ----
#         image: ndarray
#             `[H, W, 3]`, uint8 rgb.
#         hue: float
#             Shift strength of hue (HSV-H), recommended range [-1., 1.].
#         saturation: float
#             Scale strength of saturation (HSV-S), recommended range [0., 3.0].
#         brightness: float
#             Scale strength of value (HSV-V), recommended range [0.0, 2.0].
#         constract: float
#             Scale strength of constract, recommended range [0.0, 2.0].

#         """
#         assert image.shape[-1] == 3, 'Only support RGB image with 3 channels.'

#         image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32) # to hsv space
#         image[..., 0] = (image[..., 0] + 180. * hue) % 180. # hue 0~1
#         image[..., 1] = (image[..., 1] * saturation).clip(0., 255.)
#         image[..., 2] = (image[..., 2] * brightness).clip(0., 255.)
#         image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

#         image = (127.0 + (image - 127.0) * constrast).clip(0, 255) # constrast 0.0~2.0

#         return image.astype(np.uint8)


#     # -----------------------
#     @staticmethod
#     def add_guassian_noise(image:np.ndarray,
#                            noise_strength:float=10.0):

#         image = image.astype(np.float32)
#         image += np.random.normal(0, noise_strength, image.shape)
#         image = np.clip(image, 0, 255)

#         return image.astype(np.uint8)

