"""
Basedataset class for lidar data pre-processing
"""
import os
import random
from collections import OrderedDict

import PIL
import torch
import numpy as np
from torch.utils.data import Dataset

from opencood.utils.camera_utils import load_rgb_from_files
from opencood.data_utils.augmentor.data_augmentor import V2VLidarDataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, calculate_prev_pose_offset


class BaseScenarioDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to assign correct
    index. Additionally, it samples multiple frames for each scenario ().

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

    def __init__(self, params, visualize, train=True, validate=False, **kwargs):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.validate = validate

        self.pre_processor = None
        self.post_processor = None
        self.data_augmentor = V2VLidarDataAugmentor(params['data_augment'],
                                            train)
        
        if 'queue_length' in params['fusion']['args']:
            self.queue_length = max(1, params['fusion']['args']['queue_length'])
        else:
            self.queue_length = 1
        
        if 'timestamp_offset' in params['fusion']['args']:
            self.timestamp_offset = max(0, params['fusion']['args']['timestamp_offset'])
        else:
            self.timestamp_offset = 0
        
        if 'timestamp_offset_mean' in params['fusion']['args']:
            self.timestamp_offset_mean = params['fusion']['args']['timestamp_offset_mean']
            self.timestamp_offset_std = params['fusion']['args']['timestamp_offset_std']
        else:
            self.timestamp_offset_mean = 0
            self.timestamp_offset_std = 0

        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed'] \
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0

        else:
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

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        # first load all paths of different scenarios
        self.scenario_folders = sorted([os.path.join(root_dir, x)
                                        for x in os.listdir(root_dir) if
                                        os.path.isdir(
                                            os.path.join(root_dir, x))])
        self.reinitialize()

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

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
        # if the queue length is set to 1, then the vehicle offset is already the connection to the previous car
        # otherwise it starts with the second vehicle (e.g. the first vehicle offset is the identity matrix)

        # we loop the accumulated length list to see get the scenario index
        if isinstance(idx, int):
            scenario_database, timestamp_indices, prev_bev_exists, timestamp_offset = self.retrieve_by_idx(idx)
        else:
            import sys
            sys.exit('Index has to be a int')

        data_queue = []
        # Load files for all timestamps
        for i, (timestamp_index, prev_bev_exist) in enumerate(zip(timestamp_indices, prev_bev_exists)):
            # retrieve the corresponding timestamp key
            timestamp_key = self.return_timestamp_key(
                scenario_database, timestamp_index)
            # calculate distance to ego for each cav for time delay estimation
            
            ego_cav_content = [cav_content for _, cav_content in scenario_database.items() if cav_content['ego']][0]
            ego_yaml = self.load_yaml_by_timestamp(ego_cav_content, timestamp_key)

            data = OrderedDict()
            # load files for all CAVs
            for cav_id, cav_content in scenario_database.items():
                # update scenario database with distance information
                cav_yaml = self.load_yaml_by_timestamp(cav_content, timestamp_key)
                self.calc_dist_to_ego(scenario_database, ego_yaml['lidar_pose'], cav_yaml['lidar_pose'], cav_id)

                data[cav_id] = OrderedDict()
                data[cav_id]['ego'] = cav_content['ego']

                # calculate delay for this vehicle
                timestamp_delay = \
                    self.time_delay_calculation(cav_content['ego'])

                if timestamp_index - timestamp_delay <= 0:
                    timestamp_delay = timestamp_index

                timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
                timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                                timestamp_index_delay)
                # add time delay to vehicle parameters
                data[cav_id]['time_delay'] = timestamp_delay

                # load the camera transformation matrix to dictionary

                data[cav_id]['camera_params'] = \
                    self.reform_camera_param(ego_yaml, cav_yaml)
                # load the lidar params into the dictionary
                data[cav_id]['params'] = self.reform_lidar_param(
                    cav_content, ego_cav_content,
                    cav_yaml, ego_yaml,
                    timestamp_key, timestamp_key_delay, cur_ego_pose_flag)
                # todoL temporally disable pcd loading
                # data[cav_id]['lidar_np'] = \
                #     pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])

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
                
                # add previous bev information
                if i == 0:
                    data[cav_id]['prev_bev_exists'] = False  # Could also be True (see prev_bev_exists), but we keep the first always False
                    data[cav_id]['prev_pose_offset'] = np.eye(4)
                    # if queue length is set to 1, use the previous frame (if possible) to calculate the transformation matrix
                    if self.queue_length == 1:
                        # offset between current and previous frame
                        # -1 because we want the previous frame and - timestamp offset (which is the skips in between the frames)
                        prev_timestamp_index = max(0, timestamp_index_delay - timestamp_offset - 1)

                        if prev_timestamp_index >= timestamp_index_delay:
                            data[cav_id]['prev_bev_exists'] = False
                            data[cav_id]['prev_pose_offset'] = np.eye(4)
                        else:
                            prev_timestamp_key = self.return_timestamp_key(scenario_database, prev_timestamp_index)
                            prev_cav_yaml = self.load_yaml_by_timestamp(cav_content, prev_timestamp_key)
                            prev_cav_data = dict()
                            prev_cav_data['params'] = self.reform_lidar_param(
                                cav_content, ego_cav_content,
                                prev_cav_yaml, ego_yaml,
                                prev_timestamp_index, prev_timestamp_index, cur_ego_pose_flag
                            )
                            data[cav_id]['prev_bev_exists'] = True
                            data[cav_id]['prev_pose_offset'] = calculate_prev_pose_offset(data[cav_id], prev_cav_data)
                else:
                    data[cav_id]['prev_bev_exists'] = prev_bev_exist
                    data[cav_id]['prev_pose_offset'] = calculate_prev_pose_offset(
                        data[cav_id], data_queue[i - 1][cav_id])

            data_queue.append(data)

        return data_queue


    def retrieve_by_idx(self, idx):
        """
        Retrieve the scenario index and timstamp by a single idx
        .
        Parameters
        ----------
        idx : int
            Idx among all frames.
            We use the query length to get the previous (or subsequent) timestamps

        Returns
        -------
        scenario database and timestamp.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                _len_record = self.len_record[i - 1] if i > 0 else 0
                scenario_index = i
                timestamp_index = idx if i == 0 else idx - _len_record
                max_idx = self.len_record[i] - _len_record - 1
                break
        
        if self.timestamp_offset_mean > 0 and self.timestamp_offset_std > 0:
            timestamp_offset = max(0, int(np.random.normal(self.timestamp_offset_mean, self.timestamp_offset_std)))
        else:
            timestamp_offset = self.timestamp_offset
        
        # keep the timestamp indices between min_idx and max_idx
        span = (self.queue_length - 1) * timestamp_offset + self.queue_length

        if span > max_idx:
            timestamp_index = 0
            timestamp_offset = (max_idx - self.queue_length) // (self.queue_length - 1)
            span = (self.queue_length - 1) * timestamp_offset + self.queue_length

        # check if its in between min and max idx
        if span > 1:
            if (timestamp_index + span) > max_idx:
                timestamp_index = max(0, timestamp_index - span)

        scenario_database = self.scenario_database[scenario_index]

        timestamp_indices = np.array(
            [timestamp_index + i * (timestamp_offset + 1) for i in range(self.queue_length)]
        )
        
        prev_bevs_exists = []
        if timestamp_index == 0:
            prev_bevs_exists.append(False)
        else:
            prev_bevs_exists.append(True)
        
        for i in range(self.queue_length - 1):
            prev_bevs_exists.append(True)

        return scenario_database, timestamp_indices, prev_bevs_exists, timestamp_offset

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
            # make sure the path is of correct format
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
    
    def load_yaml_by_timestamp(self, cav_content, timestamp_key):
        return load_yaml(cav_content[timestamp_key]['yaml'])

    def calc_dist_to_ego(self, scenario_database, ego_lidar_pose, cav_lidar_pose, cav_id):
        """
        Calculate the distance to ego for each cav using NumPy for vectorized calculations.
        """

        assert ego_lidar_pose is not None

        # Extract lidar poses and convert them to NumPy arrays

        # Calculate distances using NumPy vectorized operations
        ego_lidar_pose = np.array(ego_lidar_pose)
        cav_lidar_pose = np.array(cav_lidar_pose)
        # the first two indices are the translation
        distance = np.linalg.norm(cav_lidar_pose[:2] - ego_lidar_pose[:2])

        # Update the scenario_database with distances
        scenario_database[cav_id]['distance_to_ego'] = distance

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == 'real':
            # noise/time is in ms unit
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == 'sim':
            time_delay = np.abs(self.async_overhead)

        # todo: current 10hz, we may consider 20hz in the future
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0],
                      pose[1] + xyz_noise[1],
                      pose[2] + xyz_noise[2],
                      pose[3],
                      pose[4] + ryp_std[1],
                      pose[5]]
        return noise_pose

    def reform_camera_param(self, ego_yaml, cav_yaml):
        """
        Load camera extrinsic and intrinsic into a propoer format. todo:
        Enable delay and localization error.

        Returns
        -------
        The camera params dictionary.
        """
        camera_params = OrderedDict()

        ego_lidar_pose = ego_yaml['lidar_pose']
        ego_pose = ego_yaml['true_ego_pos']

        # load each camera's world coordinates, extrinsic (lidar to camera)
        # pose and intrinsics (the same for all cameras).

        for i in range(4):
            camera_coords = cav_yaml['camera%d' % i]['cords']
            camera_extrinsic = np.array(
                cav_yaml['camera%d' % i]['extrinsic'])
            camera_extrinsic_to_ego_lidar = x1_to_x2(camera_coords,
                                                     ego_lidar_pose)
            camera_extrinsic_to_ego = x1_to_x2(camera_coords,
                                               ego_pose)

            camera_intrinsic = np.array(
                cav_yaml['camera%d' % i]['intrinsic'])

            cur_camera_param = {'camera_coords': camera_coords,
                                'camera_extrinsic': camera_extrinsic,
                                'camera_intrinsic': camera_intrinsic,
                                'camera_extrinsic_to_ego_lidar':
                                    camera_extrinsic_to_ego_lidar,
                                'camera_extrinsic_to_ego':
                                    camera_extrinsic_to_ego}
            camera_params.update({'camera%d' % i: cur_camera_param})

        return camera_params


    def reform_lidar_param(self, cav_content, ego_content, cav_yaml, ego_yaml, timestamp_key, timestamp_delay, cur_ego_pose_flag):
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
        if timestamp_delay == timestamp_key:
            delay_cav_params = cav_yaml
            delay_ego_params = ego_yaml
        else:
            delay_cav_params = load_yaml(cav_content[timestamp_delay]['yaml'])
            delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_cav_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params['lidar_pose']

        cur_ego_lidar_pose = ego_yaml['lidar_pose']
        cur_cav_lidar_pose = cav_yaml['lidar_pose']

        # meaning cur cav != ego (cur_ego_lidar_pose != cur_cav_lidar_pose)
        if cur_ego_lidar_pose != cur_cav_lidar_pose and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose,
                                                      self.xyz_noise_std,
                                                      self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose,
                                                    self.xyz_noise_std,
                                                    self.ryp_noise_std)

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
        delay_cav_params['vehicles'] = cav_yaml['vehicles']
        delay_cav_params['transformation_matrix'] = transformation_matrix
        delay_cav_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_cav_params['spatial_correction_matrix'] = spatial_correction_matrix

        return delay_cav_params

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

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

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
        
        # time this method
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

    def visualize_result(self, pred_box_tensor,
                         gt_tensor,
                         pcd,
                         show_vis,
                         save_path,
                         dataset=None):
        self.post_processor.visualize(pred_box_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis,
                                      save_path,
                                      dataset=dataset)