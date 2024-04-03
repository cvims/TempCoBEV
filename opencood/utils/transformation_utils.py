"""
Transformation utils
"""

import numpy as np
import math


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list or np.ndarray
        The pose of x1 under world coordinates or
        transformation matrix x1->world
    x2 : list or np.ndarray
        The pose of x2 under world coordinates or
         transformation matrix x2->world

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    if isinstance(x1, list) and isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        x2_to_world = x_to_world(x2)
        world_to_x2 = np.linalg.inv(x2_to_world)
        transformation_matrix = np.dot(world_to_x2, x1_to_world)

    # object pose is list while lidar pose is transformation matrix
    elif isinstance(x1, list) and not isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        world_to_x2 = x2
        transformation_matrix = np.dot(world_to_x2, x1_to_world)
    # both are numpy matrix
    else:
        world_to_x2 = np.linalg.inv(x2)
        transformation_matrix = np.dot(world_to_x2, x1)

    return transformation_matrix

def calculate_prev_pose_offset(cur_data, prev_data):
        """
        Calculate the transformation matrix from previous to current pose.

        Parameters
        ----------
        cur_data : dict
            Current data dictionary.

        prev_data : dict
            Previous data dictionary.

        Returns
        -------
        prev_pose_offset : np.ndarray
            Transformation matrix from previous to current pose.
        """
        if cur_data['prev_bev_exists']:
            cur_pose = cur_data['params']['lidar_pose']
            prev_pose = prev_data['params']['lidar_pose']
            prev_pose_offset = x1_to_x2(prev_pose, cur_pose)
        else:
            prev_pose_offset = np.eye(4)
        
        # # save bev_static.png to file
        # arr = cur_data['bev_static.png']
        # # to pil
        # im = PIL.Image.fromarray(arr)
        # # save
        # im.save('bev_static.png')

        # # same for prev_data
        # arr = prev_data['bev_static.png']
        # # to pil
        # im = PIL.Image.fromarray(arr)
        # # save
        # im.save('bev_static2.png')

        # # use the prev_pose_offset to translate the image
        # # first load the bev (with PIL)
        # bev = PIL.Image.open('bev_static.png')
        # # convert to np array
        # bev = np.array(bev)
        # # translate (with PIL)
        # bev = PIL.Image.fromarray(bev)

        # bev_width_in_meters = 100
        # bev_height_in_meters = 100
        # # translate in pixels

        # # x forward, y right, z up
        # # in bev image, x is horizontal, y is vertical
        # x_offset_in_pixels = prev_pose_offset[0, 3] / bev_width_in_meters * bev.size[0]
        # y_offset_in_pixels = prev_pose_offset[1, 3] / bev_height_in_meters * bev.size[1]

        # # convert to bev coordinates
        # _x_offset_in_pixels = y_offset_in_pixels
        # y_offset_in_pixels = -x_offset_in_pixels
        # x_offset_in_pixels = _x_offset_in_pixels

        # bev = PIL.ImageChops.offset(bev, int(x_offset_in_pixels), int(y_offset_in_pixels))

        # # save
        # bev.save('bev_static2_transformed.png')

        return prev_pose_offset


def dist_two_pose(cav_pose, ego_pose):
    """
    Calculate the distance between agent by given there pose.
    """
    if isinstance(cav_pose, list):
        distance = \
            math.sqrt((cav_pose[0] -
                       ego_pose[0]) ** 2 +
                      (cav_pose[1] - ego_pose[1]) ** 2)
    else:
        distance = \
            math.sqrt((cav_pose[0, -1] -
                       ego_pose[0, -1]) ** 2 +
                      (cav_pose[1, -1] - ego_pose[1, -1]) ** 2)
    return distance


def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous
