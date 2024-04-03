"""
Basedataset class for lidar data pre-processing
"""

import os
import math
import random
from collections import OrderedDict

import PIL
import torch
import numpy as np
from torch.utils.data import Dataset

import opencood.utils.pcd_utils as pcd_utils
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2

from opencood.utils import common_utils, box_utils, pcd_utils, camera_utils
from opencood.utils.camera_utils import load_rgb_from_files

from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor

import cv2


class TestBaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to assign correct
    index.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the dataset is used for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    """

    def __init__(self, params, visualize, train=True, validate=False):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.validate = validate

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = build_postprocessor(params['postprocess'], train)

        self.async_flag = False
        self.async_overhead = 0  # ms
        self.async_mode = 'sim'
        self.loc_err_flag = False
        self.xyz_noise_std = 0
        self.ryp_noise_std = 0
        self.data_size = 0  # Mb
        self.transmission_speed = 27  # Mbps
        self.backbone_delay = 0  # ms

        if self.train and not self.validate:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']

        if 'max_cav' not in params['train_params']:
            self.max_cav = 7
        else:
            self.max_cav = params['train_params']['max_cav']


        # first load all paths of different scenarios
        self.scenario_folders = sorted([os.path.join(root_dir, x)
                                        for x in os.listdir(root_dir) if
                                        os.path.isdir(
                                            os.path.join(root_dir, x))])
        
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []
        self.reinitialize()

    def __len__(self):
        return self.len_record[-1]

    def reinitialize(self):
        """
        Use this function to randomly shuffle all cav orders to augment
        training.
        """
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            if self.train and not self.validate:
                cav_list = [x for x in os.listdir(scenario_folder)
                            if os.path.isdir(
                        os.path.join(scenario_folder, x))]
                random.shuffle(cav_list)
            else:
                cav_list = sorted([x for x in os.listdir(scenario_folder)
                                   if os.path.isdir(
                        os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                # use the frame number as key, the full path as the values
                # todo currently we don't load additional metadata
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.load_camera_files(cav_path, timestamp)

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = \
                        camera_files
                    
                    # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = \
                            os.path.join(cav_path,
                                         timestamp + '_' + file_extension)

                        self.scenario_database[i][cav_id][timestamp][
                            file_extension] = file_name

                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False

    @staticmethod
    def load_camera_files(cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + '_camera0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + '_camera1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + '_camera2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + '_camera3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int or tuple
            Index given by dataloader or given scenario index and timestamp.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        if isinstance(idx, int):
            scenario_database, timestamp_index = self.retrieve_by_idx(idx)
        elif isinstance(idx, tuple):
            scenario_database = self.scenario_database[idx[0]]
            timestamp_index = idx[1]
        else:
            import sys
            sys.exit('Index has to be a int or tuple')

        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        # calculate distance to ego for each cav for time delay estimation
        ego_cav_content = \
            self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # calculate delay for this vehicle
            timestamp_delay = 0

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index

            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                            timestamp_index_delay)
            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay

            # load the camera transformation matrix to dictionary
            data[cav_id]['camera_params'] = \
                self.reform_camera_param(cav_content,
                                         ego_cav_content,
                                         timestamp_key)
            # load the lidar params into the dictionary
            data[cav_id]['params'] = self.reform_lidar_param(cav_content,
                                                             ego_cav_content,
                                                             timestamp_key,
                                                             timestamp_key_delay,
                                                             cur_ego_pose_flag)
        
            data[cav_id]['lidar_np'] = \
                pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
        
            data[cav_id]['camera_np'] = \
                    load_rgb_from_files(
                        cav_content[timestamp_key_delay]['cameras'])
            
            for file_extension in self.add_data_extension:
                # todo: currently not considering delay!
                # output should be only yaml or image
                if '.yaml' in file_extension:
                    data[cav_id][file_extension] = \
                        load_yaml(cav_content[timestamp_key][file_extension])
                else:
                    # data[cav_id][file_extension] = \
                        # cv2.imread(cav_content[timestamp_key][file_extension])
                    data[cav_id][file_extension] = np.array(PIL.Image.open(cav_content[timestamp_key][file_extension]))      

        return data


    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content['ego']:
                ego_cav_content = cav_content
                ego_lidar_pose = \
                    load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
                break

        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_database.items():
            cur_lidar_pose = \
                load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
            distance = \
                math.sqrt((cur_lidar_pose[0] -
                           ego_lidar_pose[0]) ** 2 +
                          (cur_lidar_pose[1] - ego_lidar_pose[1]) ** 2)
            cav_content['distance_to_ego'] = distance
            scenario_database.update({cav_id: cav_content})

        return ego_cav_content


    def retrieve_by_idx(self, idx):
        """
        Retrieve the scenario index and timstamp by a single idx
        .
        Parameters
        ----------
        idx : int
            Idx among all frames.

        Returns
        -------
        scenario database and timestamp.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]

        return scenario_database, timestamp_index

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            # file to linux path
            file = file.replace('\\', '/')
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key


    def reform_camera_param(self, cav_content, ego_content, timestamp):
        """
        Load camera extrinsic and intrinsic into a propoer format. todo:
        Enable delay and localization error.

        Returns
        -------
        The camera params dictionary.
        """
        camera_params = OrderedDict()

        cav_params = load_yaml(cav_content[timestamp]['yaml'])
        ego_params = load_yaml(ego_content[timestamp]['yaml'])
        ego_lidar_pose = ego_params['lidar_pose']
        ego_pose = ego_params['true_ego_pos']

        # load each camera's world coordinates, extrinsic (lidar to camera)
        # pose and intrinsics (the same for all cameras).

        for i in range(4):
            camera_coords = cav_params['camera%d' % i]['cords']
            camera_extrinsic = np.array(
                cav_params['camera%d' % i]['extrinsic'])
            camera_extrinsic_to_ego_lidar = x1_to_x2(camera_coords,
                                                     ego_lidar_pose)
            camera_extrinsic_to_ego = x1_to_x2(camera_coords,
                                               ego_pose)

            camera_intrinsic = np.array(
                cav_params['camera%d' % i]['intrinsic'])

            cur_camera_param = {'camera_coords': camera_coords,
                                'camera_extrinsic': camera_extrinsic,
                                'camera_intrinsic': camera_intrinsic,
                                'camera_extrinsic_to_ego_lidar':
                                    camera_extrinsic_to_ego_lidar,
                                'camera_extrinsic_to_ego':
                                    camera_extrinsic_to_ego}
            camera_params.update({'camera%d' % i: cur_camera_param})

        return camera_params

    def reform_lidar_param(self, cav_content, ego_content, timestamp_cur,
                           timestamp_delay, cur_ego_pose_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """
        cur_params = load_yaml(cav_content[timestamp_cur]['yaml'])
        delay_params = load_yaml(cav_content[timestamp_delay]['yaml'])

        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                 cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                            cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params['vehicles'] = cur_params['vehicles']
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

        return delay_params

    @staticmethod
    def find_ego_pose(base_data_dict):
        """
        Find the ego vehicle id and corresponding LiDAR pose from all cavs.

        Parameters
        ----------
        base_data_dict : dict
            The dictionary contains all basic information of all cavs.

        Returns
        -------
        ego vehicle id and the corresponding lidar pose.
        """

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        return ego_id, ego_lidar_pose


    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Get the selected vehicle's camera
        Parameters
        ----------
        selected_cav_base : dict
            The basic information of the selected vehicle.

        ego_pose : list
            The ego vehicle's (lidar) pose.

        Returns
        -------
        objects coordinates under ego coordinate frame and corresponding
        object ids.
        """

        # generate the bounding box(n, 7) under the ego space
        object_bbx_center_ego, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       ego_pose)
        # generate the bounding box under the cav space
        object_bbx_center_cav, object_bbx_mask_cav, _ = \
            self.post_processor.generate_object_center(
                [selected_cav_base],
                selected_cav_base['params']['lidar_pose'])

        return object_bbx_center_ego[object_bbx_mask == 1], \
               object_bbx_center_cav[object_bbx_mask_cav == 1], \
               object_ids


    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)


    def get_data_sample(self, base_data_dict):
        processed_data_dict = OrderedDict()

        ego_id, ego_lidar_pose = self.find_ego_pose(base_data_dict)

        # used to save all object coordinates under ego space
        object_stack = []
        object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            processed_data_dict[cav_id] = base_data_dict[cav_id]
            # the objects bbx position under ego and cav lidar coordinate frame
            object_bbx_ego, object_bbx_cav, object_ids = \
                self.get_item_single_car(selected_cav_base,
                                         ego_lidar_pose)

            object_stack.append(object_bbx_ego)
            object_id_stack += object_ids

            processed_data_dict[cav_id]['object_bbx_cav'] = object_bbx_cav
            processed_data_dict[cav_id]['object_id'] = object_ids
            processed_data_dict[cav_id]['bev_visibility.png'] = selected_cav_base['bev_visibility.png']
            processed_data_dict[cav_id]['bev_visibility_corp.png'] = selected_cav_base['bev_visibility_corp.png']
            processed_data_dict[cav_id]['bev_dynamic.png'] = selected_cav_base['bev_dynamic.png']

        # Object stack contains all objects that can be detected from all
        # cavs nearby under ego coordinates. We need to exclude the repititions
        object_stack = np.vstack(object_stack)

        unique_indices = []
        unique_objects = []
        unique_ids = []
        for x in set(object_id_stack):
            unique_indices.append(object_id_stack.index(x))
            unique_ids.append(x)
            unique_objects.append(object_stack[object_id_stack.index(x)])

        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((100, 7))
        mask = np.zeros(100)
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # update the ego vehicle with all objects coordinates
        processed_data_dict[ego_id]['object_bbx_ego'] = object_bbx_center
        processed_data_dict[ego_id]['object_bbx_ego_mask'] = mask
        processed_data_dict[ego_id]['object_id_ego'] = unique_ids

        return processed_data_dict


    def collate_batch(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict


def filter_by_lidar_hits(lidar_np, gt_bounding_boxes, lidar_range = None, min_lidar_hits = 25):
    lidar_np = pcd_utils.mask_points_by_range(lidar_np, lidar_range)
    lidar_np = pcd_utils.downsample_lidar_minimum([lidar_np])[0]

    corners3d = box_utils.boxes_to_corners_3d(gt_bounding_boxes, order='hwl')

    # Convert ego_lidar_points to 2D array if it's not already
    lidar_points_2d = lidar_np[:, :2]

    # Use broadcasting for element-wise comparisons
    corner_mins = corners3d[:, :, :2].min(axis=1)
    corner_maxs = corners3d[:, :, :2].max(axis=1)

    # Reshape corner_mins to match the shape of ego_lidar_points_2d for broadcasting
    corner_mins_reshaped = corner_mins[:, np.newaxis, :]
    corner_maxs_reshaped = corner_maxs[:, np.newaxis, :]

    corner_min_comparison = corner_mins_reshaped <= lidar_points_2d
    corner_max_comparison = corner_maxs_reshaped >= lidar_points_2d

    # Check if any lidar point is within any bounding box
    is_hit_array = np.logical_and.reduce(corner_min_comparison, axis=2) & np.logical_and.reduce(corner_max_comparison, axis=2)

    # Count hits per bounding box
    hit_counts = np.sum(is_hit_array, axis=1)

    # hits greater than 15
    is_hit_array = hit_counts >= min_lidar_hits

    return corners3d, is_hit_array


def create_bev_map(corners3d, lidar_range, folder_name, filename):
    L1, W1, H1, L2, W2, H2 = lidar_range
    ratio = (100 / 256)
    img_row = int((L2 - L1) / ratio)
    img_col = int((W2 - W1) / ratio)

    bev_map = np.zeros((img_row, img_col))
    bev_origin = np.array([L1, W1, H1]).reshape(1, -1)
    
    corners3d = corners3d.reshape(-1, 3)

    indices = ((corners3d[:, :3] - bev_origin) / ratio).astype(int)

    # flip y axis
    indices[:, 0] = img_row - indices[:, 0]

    # reshape into boxes
    indices = indices.reshape(-1, 8, 3)
    indices = indices[:, :4, :2]

    # clip the indices to max and min of image size
    indices = np.clip(indices, 0, [img_row, img_col])

    # draw the boxes
    for index in indices:
        # index must be of shape x, y here
        # swap x and y
        index = index[:, ::-1]
        cv2.fillPoly(bev_map, [index.astype(np.int32)], 1)

    # save bev map
    cv2.imwrite(os.path.join(folder_name, filename), bev_map * 255)


def filter_by_camera_images(gt_bounding_boxes, cam_params, camera_np):
    """
    Returns:
        corners3d: (N, 8, 3) array of corners in image coordinates
        is_hit_array: (N, ) array of bools (mask)
    """

    corners_3d_for_cam = box_utils.boxes_to_corners_3d(gt_bounding_boxes, order='hwl')    

    obj_mask = []
    for i in range(4):
        name = 'camera{}'.format(i)
        np_image = camera_np[name]
        
        intrinsics = cam_params[name]['camera_intrinsic']
        extrinsics = cam_params[name]['camera_extrinsic']
        w, h = int(cam_params[name]['camera_intrinsic'][0, 2] * 2), int(cam_params[name]['camera_intrinsic'][1, 2] * 2)

        corners_3d_copy = corners_3d_for_cam.copy()
        bbx_img_coords = camera_utils.project_3d_to_camera(
            corners_3d_copy, intrinsics, extrinsics
        )
        
        bbx_img_coords, img_coords_mask = camera_utils.filter_bbx_out_scope(
            bbx_img_coords, w, h
        )

        indices_mask = np.where(img_coords_mask)[0]

        corners_3d_copy = corners_3d_copy[img_coords_mask]

        # check if corners_3d is empty
        if corners_3d_copy.shape[0] == 0:
            # still write camera image
            cv2.imwrite(f'{name}.png', np_image)
            continue

        # apply occlusion filter
        occ_mask = camera_occlusion_filter(bbx_img_coords, w, h)

        # occ_mask only contains the masks for those that were true in iter_mask
        # thus has a different shape
        false_id_mask = indices_mask[~occ_mask]
        img_coords_mask[false_id_mask] = False

        # draw boxes to image
        bbx_img_coords = bbx_img_coords[occ_mask]
        out_image = camera_utils.draw_2d_bbx(
            np_image, bbx_img_coords, color=(0, 255, 0)
        )

        # save to disk
        cv2.imwrite(f'{name}.png', out_image)

        obj_mask.append(img_coords_mask)
    
    # stack visible corners and remove duplicates
    obj_mask = np.vstack(obj_mask)
    # and operation across all cameras
    obj_mask = np.logical_or.reduce(obj_mask, axis=0)

    return corners_3d_for_cam, obj_mask


def extract_values_within_polygon(image, poly_box):
    """
    Extract values from the image within the polygon defined by poly_box.

    Args:
        image: numpy array representing the image.
        poly_box: numpy array of shape (N, 2) containing x and y pixel coordinates of the polygon.

    Returns:
        values_within_polygon: numpy array containing the values from the image within the polygon.
    """
    # Create a mask for the polygon
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [poly_box], 1)

    # Extract values from the image within the polygon using the mask
    values_within_polygon = image[mask.astype(bool)]

    return values_within_polygon


def camera_occlusion_filter(bbx_imgs_coords, width, height):
    """
    Args:
        bbx_imgs_coords: (N, 8, 3) array of corners in image coordinates (with depth)
        width: width of image
        height: height of image
    """
    # Sort bbx_img_coords by depth and save indices
    bbx_img_coords_depth = np.min(bbx_imgs_coords[:, :, 2], axis=1)
    bbx_img_coords_depth_sorted_indices = np.argsort(bbx_img_coords_depth)

    # Build a occlusion map
    occlusion_map = np.zeros((height, width))

    # Iterate through sorted bboxes and occupy the occlusion map (if not already occupied)
    # occupied objects
    filter_mask = []
    for i in bbx_img_coords_depth_sorted_indices:
        # bbox
        bbox = bbx_imgs_coords[i,:, :2]
        bbox_points = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))

        convex_hull = cv2.convexHull(bbox_points)
        convex_hull = convex_hull.reshape(-1, 2)

        # use poly box coordinates as border for the bounding box inside the image
        # and get the indices of the pixels inside the bounding box

        values_of_occ_map = extract_values_within_polygon(occlusion_map, convex_hull)

        if np.all(values_of_occ_map):
            # The bounding box is already occupied, skip this box
            filter_mask.append(False)
        else:
            # Draw the bounding box to the occlusion map
            cv2.fillPoly(occlusion_map, [convex_hull], 1)
            cv2.imwrite('occlusion_map.png', occlusion_map * 255)

            filter_mask.append(True)

    # get back the initial order of bbx_imgs_coords before depth sorting
    filter_mask = np.array(filter_mask)
    filter_mask = filter_mask[np.argsort(bbx_img_coords_depth_sorted_indices)]

    return filter_mask.astype(bool)


if __name__ == '__main__':
    params = load_yaml(r'C:\Git_Repos\temporal_cooperative_bev\opencood\hypes_yaml\opcamera\base_camera.yaml')

    lidar_range = [-50, -50, -3, 50, 50, 1]
    min_lidar_hits = 1

    filter_by_camera = True  # otherwise filter by lidar

    opencda_dataset = TestBaseDataset(params, train=False, visualize=True)

    folder_name = 'test_vis_bev'
    original_folder = os.path.join(folder_name, 'original')
    camera_generated_folder = os.path.join(folder_name, 'camera_generated')
    lidar_generated_folder = os.path.join(folder_name, 'lidar_generated')

    store_dir = camera_generated_folder if filter_by_camera else lidar_generated_folder

    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(original_folder, exist_ok=True)

    for i in range(166, 200, 1):
        base_data = opencda_dataset.retrieve_base_data(i)

        data_sample = opencda_dataset.get_data_sample(base_data)

        all_3d_corners = []
        transformation_matrices = []

        for s_id, sample in enumerate(data_sample.values()):
            is_ego = sample['ego']

            # get the bounding boxes for the current sample
            gt_bounding_boxes = sample['object_bbx_cav']
            gt_object_ids = sample['object_id']

            if filter_by_camera:
                corners3d, filtered_corners3d_mask = filter_by_camera_images(
                    gt_bounding_boxes, sample['camera_params'], sample['camera_np']
                )
            else:
                corners3d, filtered_corners3d_mask = filter_by_lidar_hits(
                    sample['lidar_np'], gt_bounding_boxes, lidar_range, min_lidar_hits
                )
            
            # Filter out hit bounding boxes using boolean indexing
            corners3d = corners3d[filtered_corners3d_mask]

            # save image (cav pov)
            file_name = str(i) + '_ego_visibility.png' if is_ego else  f'{str(i)}_cav_{s_id}_visibility.png'
            create_bev_map(corners3d, lidar_range, store_dir, file_name)

            # store original bev map (preprocessed from cobevt)
            file_suffix = f'{str(i)}_ego' if is_ego else f'{str(i)}_cav_{s_id}'

            cv2.imwrite(os.path.join(original_folder, file_suffix + '_visibility.png'), sample['bev_visibility.png'])
            cv2.imwrite(os.path.join(original_folder, file_suffix + '_visibility_corp.png'), sample['bev_visibility_corp.png'])
            cv2.imwrite(os.path.join(original_folder, file_suffix + '_dynamic.png'), sample['bev_dynamic.png'])

            # append to all corners
            all_3d_corners.append(corners3d)

            # append transformation matrix
            transformation_matrices.append(sample['params']['transformation_matrix'])


        # create combined BEV maps for each sample
        # use the transformation matrix of the other car to transform the bounding boxes

        # shape (n_samples, n_bounding_boxes, 8, 3)

        # transformation matrices: shape (n_samples, 4, 4)
        transformation_matrices = np.stack(transformation_matrices)

        # ego cooperative bev view
        ego_perspective_3d_corners = []
        for corners_3d, transformation_matrix in zip(all_3d_corners, transformation_matrices):
            ego_perspective_3d_corners.append(
                box_utils.project_box3d(
                    corners_3d, transformation_matrix
                )
            )

        # save ego perspective bev map
        create_bev_map(np.concatenate(ego_perspective_3d_corners), lidar_range, store_dir, str(i) + '_ego_perspective_corp.png')

        # cav perspective bev view
        cav_perspective_3d_corners = []
        for c in range(1, len(all_3d_corners)):
            # from ego to current cav
            transformation_matrix = np.linalg.inv(transformation_matrices[i])
            ego_perspective_3d_corners = box_utils.project_box3d(
                all_3d_corners[0], transformation_matrix
            )
            # add to list
            cav_perspective_3d_corners.append(ego_perspective_3d_corners)

            current_t_matrix = transformation_matrices[c]

            # all others to ego and then to current cav
            for j in range(1, len(all_3d_corners)):
                # t matrix from ego to current cav
                t_matrix = np.dot(transformation_matrix[j], np.linalg.inv(current_t_matrix))
                cav_perspective_3d_corners = box_utils.project_box3d(
                    all_3d_corners[j], t_matrix
                )
                # add to list
                cav_perspective_3d_corners.append(cav_perspective_3d_corners)
            
            # save cav perspective bev map
            create_bev_map(np.stack(cav_perspective_3d_corners), lidar_range, store_dir, str(i) + f'_cav_{c}_perspective_corp.png')