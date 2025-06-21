
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from numpy import ndarray
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_rotation_6d, rotation_6d_to_matrix,
                                  euler_angles_to_matrix, matrix_to_euler_angles)
from torch import Tensor

""" ============= rotation conversion functions (packaging of Pytorch3D's API) ============= """
# ----------
def euler_2_rmt(euler: Union[Tensor, ndarray], order: str = 'XYZ') -> Union[Tensor, ndarray]:
    """
    Convert euler angles to rotation matrix

    Args:
    ----
        euler: [..., 3]
            the input euler angles
        order: str
            the order of euler angles
    Returns:
    ----
        rmt: [..., 3, 3]
            the output rotation matrix
    """
    if isinstance(euler, Tensor):
        return euler_angles_to_matrix(euler, order)
    elif isinstance(euler, ndarray):
        return euler_angles_to_matrix(torch.from_numpy(euler), order).numpy()
    else:
        raise ValueError('The input type is not supported.')

# ----------
def rmt_2_euler(rmt: Union[Tensor, ndarray], order: str = 'XYZ') -> Union[Tensor, ndarray]:
    """
    Convert rotation matrix to euler angles

    Args:
    ----
        rmt: [..., 3, 3]
            the input rotation matrix
        order: str
            the order of euler angles
    Returns:
    ----
        euler: [..., 3]
            the output euler angles
    """
    if isinstance(rmt, Tensor):
        return matrix_to_euler_angles(rmt, order)
    elif isinstance(rmt, ndarray):
        return matrix_to_euler_angles(torch.from_numpy(rmt), order).numpy()
    else:
        raise ValueError('The input type is not supported.')

# ----------
def rvc_2_rmt(rvc: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """
    Convert rotation vector (axis angle) to rotation matrix

    Args:
    ----
        rvc: [..., 3]
            the input rotation vector (axis angle)
    Returns:
    ----
        rmt: [..., 3, 3]
            the output rotation matrix
    """
    if isinstance(rvc, Tensor):
        return axis_angle_to_matrix(rvc)
    elif isinstance(rvc, ndarray):
        return axis_angle_to_matrix(torch.from_numpy(rvc)).numpy()
    else:
        raise ValueError('The input type is not supported.')


# ----------
def rmt_2_rvc(rmt: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """
    Convert rotation matrix to rotation vector (axis angle)

    Args:
    ----
        rmt: [..., 3, 3]
            the input rotation matrix
    Returns:
    ----
        rvc: [..., 3]
            the output rotation vector (axis angle)
    """
    if isinstance(rmt, Tensor):
        return matrix_to_axis_angle(rmt)
    elif isinstance(rmt, ndarray):
        return matrix_to_axis_angle(torch.from_numpy(rmt)).numpy()
    else:
        raise ValueError('The input type is not supported.')


# ----------
def r6d_2_rmt(r6d: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """
    Convert rotation 6d to rotation matrix

    Args:
    ----
        r6d: [..., 6]
            the input rotation 6d
    Returns:
    ----
        rmt: [..., 3, 3]
            the output rotation matrix
    """
    if isinstance(r6d, Tensor):
        return rotation_6d_to_matrix(r6d)
    elif isinstance(r6d, ndarray):
        return rotation_6d_to_matrix(torch.from_numpy(r6d)).numpy()
    else:
        raise ValueError('The input type is not supported.')


# ----------
def rmt_2_r6d(rmt: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """
    Convert rotation matrix to rotation 6d

    Args:
    ----
        rmt: [..., 3, 3]
            the input rotation matrix
    Returns:
    ----
        r6d: [..., 6]
            the output rotation 6d
    """
    if isinstance(rmt, Tensor):
        return matrix_to_rotation_6d(rmt)
    elif isinstance(rmt, ndarray):
        return matrix_to_rotation_6d(torch.from_numpy(rmt)).numpy()
    else:
        raise ValueError('The input type is not supported.')












""" ============= data augmentation functions ============= """

# -----------------------
def produce_points2d_bbox(pts2d:ndarray, expansion:float=1.0):
    """
    Produce 2d bounding box from 2d points.

    Parameters
    ----
        pts2d: ndarray
            `[N, 2]`, 2d points.
        expansion: float
            the expansion ratio of the bounding box.
    """
    xs, ys = pts2d[:,0], pts2d[:,1]
    xmin, ymin = min(xs), min(ys)
    xmax, ymax = max(xs), max(ys)

    x_center = (xmin + xmax) / 2.
    width = (xmax - xmin) * expansion
    y_center = (ymin + ymax) / 2.
    height = (ymax - ymin) * expansion

    edge = max(width, height)
    xmin = x_center - 0.5 * edge
    ymin = y_center - 0.5 * edge

    bbox = np.array([xmin, ymin, edge, edge]).astype(np.float32)

    return bbox


# -----------------
def compute_bbox_M(bbox:Union[ndarray, List], # [topleft_x, topleft_y, len_x, len_y]
                   dst_size:List # [W, H]
                   ) -> ndarray:
    """
    Compute the crop and resize transformation matrix.

    Parameters
    ----
        bbox: ndarray, List
            `[topleft_x, topleft_y, len_x, len_y]`
        dst_size: List
            `[W, H]`, target size.
    """

    translate = translate2d(-bbox[0], -bbox[1])
    scale = scale2d(dst_size[0] / bbox[2], dst_size[1] / bbox[3])
    bbox_M = scale @ translate

    return bbox_M


# -----------------
def apply_crop_resize(image:ndarray,
                      bbox:Union[ndarray, List], # [topleft_x, topleft_y, len_x, len_y]
                      dst_size:List, # [W, H]
                      ) -> ndarray:
    """ Apply crop and resize transformation on uint8 ndarray image.

    Parameters
    ----
        image: ndarray
            `[H, W, C]`, uint8 rgb.
        bbox: ndarray, List
            `[topleft_x, topleft_y, len_x, len_y]`
        dst_size: List
            `[W, H]`, target size.

    Returns
    ----
        crop_image: ndarray
            `[h,w,c]`
        crop_M: ndarray
            the crop and resize transformation matrix.
    """

    translate = translate2d(-bbox[0], -bbox[1])
    scale = scale2d(dst_size[0] / bbox[2], dst_size[1] / bbox[3])
    crop_M = scale @ translate

    image = cv2.warpAffine(image, crop_M[:2, :],
                            dsize=dst_size,
                            flags=cv2.INTER_LINEAR)
    return image, crop_M


## ------------
def transform2d(xy:ndarray, matrix:ndarray):
    """
    Applied 2d transformation on 2d points.
    matrix: [2, 3] or [3, 3].
    """
    x = xy[:,0] * matrix[0,0] + xy[:,1] * matrix[0,1] + matrix[0,2]
    y = xy[:,0] * matrix[1,0] + xy[:,1] * matrix[1,1] + matrix[1,2]

    return np.stack([x, y], axis=-1)


## ------------
def rotate2d(rad:float): # rotation angles in radians
    cos, sin = np.cos(rad), np.sin(rad)
    return np.array([[cos, -sin, 0],
                     [sin,  cos, 0],
                     [  0,    0, 1],], dtype=np.float32)

## ------------
def translate2d(tx:float, ty:float):
    return np.array([[1., 0., tx],
                     [0., 1., ty],
                     [0., 0., 1.],], dtype=np.float32)

## ------------
def scale2d(sx:float, sy:float):
    return np.array([[sx, 0., 0.],
                     [0., sy, 0.],
                     [0., 0., 1.],], dtype=np.float32)

## ------------
def flip2d(H:int=None, W:int=None):  # if flip around H (Y axis), input W, if flip on H,W together, input H,W together
    M = np.eye(3)
    if W is not None:
        M[0, 0], M[0, 2] = -1, W
    if H is not None:
        M[1, 1], M[1, 2] = -1, H
    return M




## ===================
class AugmentPipe:
    def __init__(self,
                 color_prob:float=1.0,
                 noise_prob:float=1.0,
                 geometry_prob:float=1.0) -> None:

        self.color_prob = color_prob
        self.noise_prob = noise_prob
        self.geometry_prob = geometry_prob

    # --------------------
    def get_augment_params_randomly(self, discard_rotation:bool=False) -> Dict:
        """ Get augment parameters randomly."""
        do_color_aug = np.random.rand(4) < self.color_prob
        do_geome_aug = np.random.rand(3) < self.geometry_prob
        do_noise_aug = np.random.rand(1) < self.noise_prob

        color_strengths = [0.05, 0.25, 0.5, 0.2]
        color_aug_values = np.random.normal(0.0, 0.4, 4).astype(np.float32).clip(-1., 1.)
        hue = do_color_aug[0] * color_aug_values[0] * color_strengths[0]
        sat = 1.0 + do_color_aug[1] * color_aug_values[1] * color_strengths[1]
        bri = 1.0 + do_color_aug[2] * color_aug_values[2] * color_strengths[2]
        con = 1.0 + do_color_aug[3] * color_aug_values[3] * color_strengths[3]

        geome_strengths = [0.15, 1.0, 0.125] # scale, rotate, tranlation
        geome_aug_values = np.random.normal(0.0, 0.4, 4).astype(np.float32).clip(-1., 1.)
        scale   = 1.0 - (do_geome_aug[0] * geome_aug_values[0] * geome_strengths[0])
        rotate  = do_geome_aug[1] * geome_aug_values[1] * geome_strengths[1]
        trans_x = do_geome_aug[2] * geome_aug_values[2] * geome_strengths[2]
        trans_y = do_geome_aug[2] * geome_aug_values[3] * geome_strengths[2]

        if discard_rotation:
            rotate = rotate * 0.0

        noise_strength = np.random.rand(1) * 10.0

        return dict(color_args=(hue, sat, bri, con),
                    geome_args=(scale, rotate, [trans_x, trans_y]),
                    noise_args=(do_noise_aug, noise_strength))

    # --------------------
    def forward_augment(self, augment_params:Dict, image:ndarray, j2d:ndarray=None, j3d:ndarray=None, rootrot:ndarray=None):

        image = self.apply_color_transform(image, *augment_params['color_args']) if self.color_prob != 0 else image
        image, (S, R, T, M) = self.apply_geometry_transform(image, *augment_params['geome_args'])
        image = self.add_guassian_noise(image, *augment_params['noise_args'])

        j2d = transform2d(j2d, M) if j2d is not None else None
        j3d = j3d @ R.T if j3d is not None else None
        rootrot = R @ rootrot if rootrot is not None else None

        return image, (j2d, j3d, rootrot), M

    # --------------------
    def forward(self, image:ndarray, j2d:ndarray=None, j3d:ndarray=None, rootrot:ndarray=None, **kwargs):
        """ Augment the data randomly.

        Parameters
        ----
        image: ndarray
            `[H, W, C]`, uint8 rgb.
        j2d: ndarray
            `[N, 2]`
        j3d: ndarray
            `[N, 3]`
        rootrot: ndarray
            `[3, 3]`, global rotation.
        M: ndarray
            '[3, 3]', the integrated transformation matrix.
        """
        aug_params = self.get_augment_params_randomly()
        return self.forward_augment(aug_params, image, j2d, j3d, rootrot)

    # --------------------
    @staticmethod
    def apply_geometry_transform(image:np.ndarray,
                                 scale:float=1.0,
                                 rotation:float=0.0, # clockwise, in radiance
                                 translation:Union[ndarray, List]=[0.0, 0.0], # [H, W]
                                 ) -> ndarray:
        """Apply geometric transformation with the image center as the origin.

        Parameters
        ----
        image: ndarray
            `[H, W, C]`, uint8 rgb.
        scale: float
            scale of image.
        rotation: float
            rotation of image. in radiance and clockwise
        translation: list
            translation of image. [H, W], ratio of image resolution. +X right, +Y down.

        Returns:
        ----
        affine_M: ndarray
            the integrated transformation matrix.

        """
        H, W, _ = image.shape

        origin = [W/2, H/2]
        trans_0 = [-origin[0], -origin[1]]
        trans_1 = [origin[0], origin[1]]

        S = scale2d(scale, scale)
        R = rotate2d(rotation)
        T = translate2d(translation[1]*W, translation[0]*H)
        M = translate2d(*trans_1) @ T @ R @ S @ translate2d(*trans_0) # stay the center. scale -> rot -> trans

        image = cv2.warpAffine(image, M[:2, :],
                               dsize=[W, H],
                               flags=cv2.INTER_LINEAR,)
                            #    borderMode=cv2.BORDER_CONSTANT,
                            #    borderValue=[124, 116, 103]) # the mean of imagenet dataset.

        return image, (S, R, T, M)

    # ---------------------
    @staticmethod
    def apply_color_transform(image:np.ndarray,
                              hue:float=0.0,
                              saturation:float=1.0,
                              brightness:float=1.0,
                              constrast:float=1.0,
                              ) -> ndarray:
        """Apply color transformation on uint8 ndarray image.

        Parameters
        ----
        image: ndarray
            `[H, W, 3]`, uint8 rgb.
        hue: float
            Shift strength of hue (HSV-H), recommended range [-1., 1.].
        saturation: float
            Scale strength of saturation (HSV-S), recommended range [0., 3.0].
        brightness: float
            Scale strength of value (HSV-V), recommended range [0.0, 2.0].
        constract: float
            Scale strength of constract, recommended range [0.0, 2.0].

        """
        assert image.shape[-1] == 3, 'Only support RGB image with 3 channels.'

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32) # to hsv space
        image[..., 0] = (image[..., 0] + 180. * hue) % 180. # hue 0~1
        image[..., 1] = (image[..., 1] * saturation).clip(0., 255.)
        image[..., 2] = (image[..., 2] * brightness).clip(0., 255.)
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        image = (127.0 + (image - 127.0) * constrast).clip(0, 255) # constrast 0.0~2.0

        return image.astype(np.uint8)

    # -----------------------
    @staticmethod
    def add_guassian_noise(image:np.ndarray,
                           do_noise:bool=False,
                           noise_strength:float=10.0):
        """Add guassian noise on image."""

        if not do_noise:
            return image

        image = image.astype(np.float32)
        image += np.random.normal(0, noise_strength, image.shape)
        image = np.clip(image, 0, 255)

        return image.astype(np.uint8)




# -----------------------
def compute_optimal_projection_params(point3d: np.ndarray, point2d: np.ndarray,
                                      use_translation=True, iterations=5):
    """
    Compute the optimal projection matrix (K) and translation vector to minimize the reprojection error from 3D points to 2D projections.

    Parameters:
    ----
        point3d (np.ndarray): An array of shape (N, 3) containing 3D points.
        point2d (np.ndarray): An array of shape (N, 2) containing corresponding 2D projections.
        use_translation (bool): Whether to use an additional translation vector for optimization.
        iterations (int): Number of iterations for optimization.

    Returns:
    ----
        proj_K (np.ndarray): A projection matrix of shape (3, 3).
        translation (np.ndarray or None): A translation vector of shape (3,) if use_translation is True, otherwise None.
        reproj_err (float): The reprojection error after optimization.
    """

    assert point3d.shape[0] == point2d.shape[0], "The number of 3D points and 2D points must match"
    assert point3d.shape[1] == 3 and point2d.shape[1] == 2, "Point dimensions must be correct"

    num_points = point3d.shape[0]

    if use_translation:
        assert point3d.shape[0] >= 6, "At least 6 points are required to compute K and translation"

        # initialize parameters
        fx = fy = max(point2d[:, 0].max(), point2d[:, 1].max()) * 1.1
        cx, cy = point2d[:, 0].mean(), point2d[:, 1].mean()
        tx = ty = tz = 0.0

        # NOTE do multi-iteration to refine the parameters, generally 3-5 iterations are enough
        for _ in range(iterations):
            # Set up the system of equations: A @ params = b
            A = np.zeros((2 * num_points, 7))  # fx, fy, cx, cy, tx, ty, tz
            b = np.zeros(2 * num_points)

            X, Y, Z = point3d[:, 0], point3d[:, 1], point3d[:, 2]
            X_t, Y_t, Z_t = X + tx, Y + ty, Z + tz

            x_proj = fx * X_t / Z_t + cx
            y_proj = fy * Y_t / Z_t + cy

            # x coordinate equation
            idx_x = np.arange(0, 2 * num_points, 2)
            A[idx_x, 0] = X_t / Z_t  # fx partial derivative
            A[idx_x, 2] = 1.0        # cx partial derivative
            A[idx_x, 4] = fx / Z_t   # tx partial derivative
            A[idx_x, 6] = -fx * X_t / (Z_t * Z_t)  # tz partial derivative
            b[idx_x] = point2d[:, 0] - x_proj

            # y coordinate equation
            idx_y = np.arange(1, 2 * num_points, 2)
            A[idx_y, 1] = Y_t / Z_t  # fy partial derivative
            A[idx_y, 3] = 1.0        # cy partial derivative
            A[idx_y, 5] = fy / Z_t   # ty partial derivative
            A[idx_y, 6] = -fy * Y_t / (Z_t * Z_t)  # tz partial derivative
            b[idx_y] = point2d[:, 1] - y_proj

            # Add regularization to avoid extreme values
            reg_scale = 1e-5  # strength of regularization
            A_reg = np.eye(7) * reg_scale
            b_reg = np.zeros(7)
            A_full = np.vstack([A, A_reg])
            b_full = np.concatenate([b, b_reg])

            # Solve for parameter adjustments
            adjustments, residuals, rank, s = np.linalg.lstsq(A_full, b_full, rcond=None)

            # Apply adjustments
            fx += adjustments[0]
            fy += adjustments[1]
            cx += adjustments[2]
            cy += adjustments[3]
            tx += adjustments[4]
            ty += adjustments[5]
            tz += adjustments[6]

        proj_K = np.array([[fx, 0, cx], [0, fy, cy], [0,  0,  1]], dtype=np.float32)
        translation = np.array([tx, ty, tz], dtype=np.float32)

        reproj_err = check_reprojection_error(point3d, point2d, proj_K, translation)

        return proj_K, translation, reproj_err

    else:
        fx = fy = max(point2d[:, 0].max(), point2d[:, 1].max()) * 1.2
        cx, cy = point2d[:, 0].mean(), point2d[:, 1].mean()

        for _ in range(iterations):
            X, Y, Z = point3d[:, 0], point3d[:, 1], point3d[:, 2]

            x_proj = fx * X / Z + cx
            y_proj = fy * Y / Z + cy

            # Set up the system of equations: A @ params = b
            A = np.zeros((2 * num_points, 4))  # fx, fy, cx, cy
            b = np.zeros(2 * num_points)

            # x coordinate equation
            idx_x = np.arange(0, 2 * num_points, 2)
            A[idx_x, 0] = X / Z  # fx偏导
            A[idx_x, 2] = 1.0    # cx偏导
            b[idx_x] = point2d[:, 0] - x_proj

            # y coordinate equation
            idx_y = np.arange(1, 2 * num_points, 2)
            A[idx_y, 1] = Y / Z  # fy偏导
            A[idx_y, 3] = 1.0    # cy偏导
            b[idx_y] = point2d[:, 1] - y_proj

            # Add regularization
            reg_scale = 1e-5
            A_reg = np.eye(4) * reg_scale
            b_reg = np.zeros(4)

            A_full = np.vstack([A, A_reg])
            b_full = np.concatenate([b, b_reg])

            # Apply adjustments
            adjustments, residuals, rank, s = np.linalg.lstsq(A_full, b_full, rcond=None)
            fx += adjustments[0]
            fy += adjustments[1]
            cx += adjustments[2]
            cy += adjustments[3]

        proj_K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        reproj_err = check_reprojection_error(point3d, point2d, proj_K)

        return proj_K, None, reproj_err

# -----------------------
def check_reprojection_error(point3d, point2d, proj_K, translation=None):
    """Check reprojection error.
    """
    if translation is not None:
        point3d_t = point3d + translation
    else:
        point3d_t = point3d
    proj_2d = ((point3d_t / point3d_t[:, 2:]) @ proj_K.T)[:, :2]
    return np.linalg.norm(proj_2d - point2d, axis=1).mean()