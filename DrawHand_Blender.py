import bpy
from mathutils import Vector
import numpy as np
from typing import List, Tuple
import math
import random


"""!!!!!!! Need run in English Language Environment !!!!!!"""

# Define the skeleton structure and colors
SKELETON = [[0, 1],  [1, 2],  [2,  3],  [3,  4],
            [0, 5],  [5, 6],  [6,  7],  [7,  8],
            [0, 9],  [9, 10], [10, 11], [11, 12],
            [0, 13], [13,14], [14, 15], [15, 16],
            [0, 17], [17,18], [18, 19], [19, 20]]



#SKELETON_RGB = [[  0, 134, 139], [  0, 134, 139], [  0, 134, 139], [  0, 134, 139],
#                [ 46, 139,  87], [ 46, 139,  87], [ 46, 139,  87], [ 46, 139,  87],
#                [139, 117,   0], [139, 117,   0], [139, 117,   0], [139, 117,   0],
#                [139,  76,  57], [139,  76,  57], [139,  76,  57], [139,  76,  57],
#                [139,  71, 137], [139,  71, 137], [139,  71, 137], [139,  71, 137]]

SKELETON_RGB = [[211,  44,  31], [211,  44,  31], [211,  44,  31], [211,  44,  31],
                [205, 140, 149], [205, 140, 149], [205, 140, 149], [205, 140, 149],
                [ 67, 107, 173], [ 67, 107, 173], [ 67, 107, 173], [ 67, 107, 173],
                [205, 173,   0], [205, 173,   0], [205, 173,   0], [205, 173,   0],
                [  4, 244, 137], [  4, 244, 137], [  4, 244, 137], [  4, 244, 137]]


# ------------------------
def create_mesh_object(verts: np.ndarray, faces: np.ndarray, color: Tuple[float, float, float, float], name: str='Mesh_Object') -> bpy.types.Object:
    """Create a new scene object from the given vertices and faces"""

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)

    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()

    mat = bpy.data.materials.new(name=f'{name}_mat')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Alpha'].default_value = min(color[-1], 1.0)
    if bsdf.inputs['Alpha'].default_value < 0.999:
        mat.blend_method = "BLEND"
    else:
        mat.blend_method = "OPAQUE"
    mat.use_backface_culling = True # backface culling, to avoid shading confusion when render in transparency with "BLEND" mode.
    obj.data.materials.append(mat)

    return obj


# -------------------------
def create_camera_from_K(K, width, height, camera_name='Camera'):
    """
    Create a new camera in Blender and set its parameters to match those given in the
    3x3 matrix K. Adjust the camera position and orientation according to the specified
    coordinate system.

    NOTE: the camera coordinate is +X right, +Y down, +Z in,
    """
    # Normalize the intrinsic matrix
    K = K / K[2, 2]

    fx = K[0, 0]
    fy = K[1, 1]
    f = (fx + fy) / 2

    # Assume the sensor width is 32mm (standard full-frame sensor)
    sensor_width = 32

    # Calculate the focal length in mm
    focal_length = f / max(width, height) * sensor_width

    # Create a new camera
    bpy.ops.object.camera_add(location=(0, 0, 0))

    # Get a reference to the newly created camera
    camera = bpy.context.object
    camera.name = camera_name
#    camera.rotation_euler = (math.pi, 0, 0) # for +Z in

    # Set the camera parameters
    camera.data.lens = focal_length
    camera.data.sensor_width = sensor_width
    camera.data.shift_x = (width / 2 - K[0, 2]) / width         # for +X right
    camera.data.shift_y = (height / 2 - K[1, 2]) / height * -1  # for +Y down
    camera.rotation_euler = (math.pi, 0, 0)                     # for +Z in

    # Set the new camera as the active camera
    bpy.context.scene.camera = camera


# --------------------------
def clear_scene():
    """Clear all Meshes, Materials, and Cameras from the scene."""
    bpy.ops.object.select_all(action='DESELECT')
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)



# -------------------------
def create_material(name: str, color: Tuple[float, float, float, float]) -> bpy.types.Material:
    """Create a new material and set its color"""
    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = color
    return mat



# -------------------------
def create_bone(joint_start: Vector, joint_end: Vector, radius: float, name: str, color: Tuple[float, float, float, float] = (1, 1, 1, 1)) -> bpy.types.Object:
    """Create a new bone and set its position, rotation, and material"""
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=(joint_start-joint_end).length, enter_editmode=False)
    bone = bpy.context.active_object
    bone.name = name
    bone.location = (joint_start + joint_end) / 2
    bone.rotation_mode = 'QUATERNION'
    bone.rotation_quaternion = (joint_end - joint_start).to_track_quat('Z', 'Y')
    mat = create_material(f'{name}_mat', color)
    bone.data.materials.append(mat)
    return bone



# -------------------------
def create_joint(coord: Vector, radius: float, name: str, color: Tuple[float, float, float, float] = (1, 1, 1, 1)) -> bpy.types.Object:
    """Create a new joint and set its position and material"""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, enter_editmode=False)
    joint = bpy.context.active_object
    joint.name = name
    joint.location = coord
    mat = create_material(f'{name}_mat', color)
    joint.data.materials.append(mat)
    return joint



# -------------------------
def create_hand_skeleton(joint_coords: List[List[float]],
                         name: str='Hand_Skeleton',
                         bone_size: float = 0.001, # in meters
                         joint_size: float = 0.001,
                         ) -> Tuple[List[bpy.types.Object], List[bpy.types.Object]]:
    """Create a hand skeleton from given joint coordinates"""

    bones = []
    joints = []
    for i, (start, end) in enumerate(SKELETON):
        joint_start = Vector(joint_coords[start])
        joint_end = Vector(joint_coords[end])
        color = tuple(c / 255 for c in SKELETON_RGB[i]) + (1,)
        bone = create_bone(joint_start, joint_end, bone_size, f'Bone_{i}', color=color)
        joint = create_joint(joint_end, joint_size, f'Joint_{start}', color=color)
        bones.append(bone)
        joints.append(joint)


    for obj in bones + joints:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = bones[0]
    bpy.ops.object.join()
    bpy.context.object.name = name

    return bones, joints




