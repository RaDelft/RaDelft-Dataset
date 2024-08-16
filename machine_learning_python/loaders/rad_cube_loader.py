import sys

import torch

# apend the absolute path of the parent directory
sys.path.append(sys.path[0] + "/..")
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io
from data_preparation import data_preparation
import numpy as np
from loaders.TopicLoader import TopicIndex

class RADCUBE_DATASET(Dataset):
    """
    Data Loader for the RaDelf dataset. It initialises a dictionary with the paths to the files of the radar
    camera and lidar.
    This is the version for single frame as input, no temporal information.

    Attributes:
        mode: train, val or test
        params: a dictionary with the parameters defined in data_preparation.py
    """
    def __init__(self, mode='train', params=None):

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise ValueError("mode should be either train, val or test")

        self.dataset_path = params['dataset_path']
        self.train_val_scenes = params['train_val_scenes']
        self.test_scenes = params['test_scenes']

        self.params = params

        # files are named as ELE_Frame_xxx and Pow_Frame_xxx. Lets get all files that matches these
        if mode == 'train' or mode == 'val':
            scene_set = self.train_val_scenes
        else:
            scene_set = self.test_scenes

        # make a dictionary, indices are keys, elevation, power, and gt_paths are values
        self.data_dict = {}
        global_array_index = 0
        # IMPORTANT: It is assumed that the folders structure is as given in the dataset. If the folder
        # structure is changed this will not work.
        for scene_number in scene_set:

            # Here it is assumed the folders structure is as given in the dataset.
            # If modified, this lines have to be changed, specially "Scene" "and RadarCubes"
            scene_dir = self.dataset_path + '/Scene' + str(scene_number)
            cubes_dir = scene_dir + '/RadarCubes'
            all_files = os.listdir(cubes_dir)
            power_files = [file for file in all_files if "Pow_Frame" in file]
            power_numbers = [int(file.split("_")[-1].split(".")[0]) for file in power_files]

            power_numbers.sort()
            indices = power_numbers.copy()
            indices = np.array(indices)

            # if train: Take 9 indices and skip one. 90% training in the train_val dataset
            if mode == 'train':
                reminder = len(indices) % 10

                if reminder != 0:
                    indices_aux = indices[:-reminder]
                    indices_aux = indices_aux.reshape(-1, 10)[:, :9].reshape(-1)
                    indices = np.concatenate([indices_aux, indices[-reminder:]])
                else:
                    indices = indices.reshape(-1, 10)[:, :9].reshape(-1)

            # if val: Skip 9 indices and take the 10th. 10% val in the train_val dataset
            elif mode == 'val':
                reminder = len(indices) % 10
                if reminder != 0:
                    indices = indices[:-reminder]
                indices = indices.reshape(-1, 10)[:, -1].reshape(-1)

            # if test we keep all the indices

            # get timestamp mapping
            timestamps_path = cubes_dir + '/timestamps.mat'
            frame_num_to_timestamp = scipy.io.loadmat(timestamps_path)
            frame_num_to_timestamp = frame_num_to_timestamp["unixDateTime"]

            rosDS_path = scene_dir + '/rosDS'
            lidar_path = rosDS_path + '/rslidar_points_clean'
            camera_dir = rosDS_path + '/ueye_left_image_rect_color'
            if params['cfar_folder'] is not None:
                cfar_dir = rosDS_path + '/' + params['cfar_folder']

            # get lidar timestamps
            lidar_timestamps_and_paths = data_preparation.get_timestamps_and_paths(lidar_path)
            camera_timestamps_and_paths = data_preparation.get_timestamps_and_paths(camera_dir)
            if params['cfar_folder'] is not None:
                cfar_timestamps_and_paths = data_preparation.get_timestamps_and_paths(cfar_dir)
            for index in indices:
                self.data_dict[global_array_index] = {}

                ## handle radar
                self.data_dict[global_array_index]["elevation_path"] = os.path.join(cubes_dir,
                                                                                    "Ele_Frame_" + str(index) + ".mat")
                self.data_dict[global_array_index]["power_path"] = os.path.join(cubes_dir,
                                                                                "Pow_Frame_" + str(index) + ".mat")

                self.data_dict[global_array_index]["timestamp"] = (frame_num_to_timestamp[index - 1][0]) * 10 ** 9
                self.data_dict[global_array_index]["numpy_cube_path"] = os.path.join(cubes_dir,
                                                                                     "radar_cube_" + str(
                                                                                         index) + ".npy")
                ## handle LiDAR
                closest_lidar_time = data_preparation.closest_timestamp(self.data_dict[global_array_index]["timestamp"],
                                                                        lidar_timestamps_and_paths)
                self.data_dict[global_array_index]["gt_path"] = lidar_timestamps_and_paths[closest_lidar_time]
                self.data_dict[global_array_index]["gt_timestamp"] = closest_lidar_time

                ## handle camera
                closest_cam_time = data_preparation.closest_timestamp(self.data_dict[global_array_index]["timestamp"],
                                                                      camera_timestamps_and_paths)
                self.data_dict[global_array_index]["cam_path"] = camera_timestamps_and_paths[closest_cam_time]
                self.data_dict[global_array_index]["cam_timestamp"] = closest_cam_time

                ## handle CFAR
                if params['cfar_folder'] is not None:
                    closest_cfar_time = data_preparation.closest_timestamp(
                        self.data_dict[global_array_index]["timestamp"],
                        cfar_timestamps_and_paths)
                    self.data_dict[global_array_index]["cfar_path"] = cfar_timestamps_and_paths[closest_cfar_time]
                    self.data_dict[global_array_index]["cfar_timestamp"] = closest_cfar_time

                global_array_index = global_array_index + 1

        # print division line
        print("--------------------------------------------------")

        print(mode + " dataset loaded with " + str(len(self.data_dict)) + " samples")
        print("scenes used: " + str(scene_set))
        # print division line
        print("--------------------------------------------------")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):

        # load elevation and power
        if not self.params['bev']:
            elevation = scipy.io.loadmat(self.data_dict[idx]["elevation_path"])["elevationIndex"]
            elevation = elevation.astype(np.single)
            elevation = np.nan_to_num(elevation, nan=17.0)
            elevation = elevation / 34

        power = scipy.io.loadmat(self.data_dict[idx]["power_path"])["radarCube"]
        power = power.astype(np.single)
        # Hardcoded maximum value after data exploration
        power = power / 8998.5576
        # combine them into a single cube with 2 channels
        if not self.params['bev']:
            input_cube = np.stack((power, elevation))
        else:
            input_cube = power
        # load gt
        gt_cloud = data_preparation.read_pointcloud(self.data_dict[idx]["gt_path"], mode="rs_lidar_clean")

        item_params = self.data_dict[idx]  # this is a dictionary with all the paths and timestamps
        gt_cube = data_preparation.lidarpc_to_lidarcube(gt_cloud, self.params)

        if not self.params['bev']:
            zero_pad = np.zeros([2, 12, 128, 240], dtype='single')
            input_cube = np.concatenate([input_cube, zero_pad], axis=1)
            zero_pad = np.zeros([2, 512, 128, 8], dtype='single')
            input_cube = np.concatenate([zero_pad, input_cube, zero_pad], axis=3)
            input_cube = np.transpose(input_cube, (0, 2, 1, 3))  # (C, H, W)

        else:
            zero_pad = np.zeros([12, 128, 240], dtype='single')
            input_cube = np.concatenate([input_cube, zero_pad])
            zero_pad = np.zeros([512, 128, 8], dtype='single')
            input_cube = np.concatenate([zero_pad, input_cube, zero_pad], 2)
            input_cube = np.transpose(input_cube, (1, 0, 2))  # (C, H, W)

        return input_cube, gt_cube, item_params


class RADCUBE_DATASET_TIME(Dataset):
    """
      Data Loader for the RaDelf dataset. It initialises a dictionary with the paths to the files of the radar
      camera and lidar.
      This is the version for multi frame as input, with temporal information.

      Attributes:
          mode: train, val or test
          params: a dictionary with the parameters defined in data_preparation.py
      """
    def __init__(self, mode='train', params=None):

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise ValueError("mode should be either train, val or test")

        self.dataset_path = params['dataset_path']
        self.train_val_scenes = params['train_val_scenes']
        self.test_scenes = params['test_scenes']

        self.params = params

        window_size = 3  # ToDo: this is not generic, if you change the number it wont work

        # files are named as ELE_Frame_xxx and Pow_Frame_xxx. Lets get all files that matches these
        if mode == 'train' or mode == 'val':
            scene_set = self.train_val_scenes
        else:
            scene_set = self.test_scenes

        # make a dictionary, indices are keys, elevation, power, and gt_paths are values
        self.data_dict = {}
        global_array_index = 0
        aux_data_dict = {}
        aux_array_index = 0

        # IMPORTANT: It is assumed that the folders structure is as given in the dataset. If the folder
        # structure is changed this will not work.
        for scene_number in scene_set:

            scene_dir = self.dataset_path + '/Scene' + str(scene_number)
            cubes_dir = scene_dir + '/RadarCubes'
            all_files = os.listdir(cubes_dir)
            power_files = [file for file in all_files if "Pow_Frame" in file]
            power_numbers = [int(file.split("_")[-1].split(".")[0]) for file in power_files]

            power_numbers.sort()
            indices = power_numbers.copy()
            indices = np.array(indices)
            total_num_samples = len(indices)

            # if train: Take 9 indices and skip one. 90% training in the train_val dataset
            if mode == 'train':
                reminder = len(indices) % 30

                # We are dropping some samples here to make it multiple of the window
                if reminder != 0:
                    indices_aux = indices[:-reminder]
                    indices_aux = indices_aux.reshape(-1, 30)[:, :27].reshape(-1)
                    indices = indices_aux
                else:
                    indices = indices.reshape(-1, 30)[:, :27].reshape(-1)

            # if val: Skip 9 indices and take the 10th. 10% val in the train_val dataset
            elif mode == 'val':
                reminder = len(indices) % 30
                # We are dropping some samples here to make it multiple of the window
                if reminder != 0:
                    indices = indices[:-reminder]
                indices = indices.reshape(-1, 30)[:, -3:].reshape(-1)

            # if test we make it divisable by the window
            elif mode == 'test':
                reminder = len(indices) % window_size
                # We are dropping some samples here to make it multiple of the window
                if reminder != 0:
                    indices = indices[:-reminder]

            #indices = indices[0:57]

            # get timestamp mapping
            timestamps_path = cubes_dir + '/timestamps.mat'
            frame_num_to_timestamp = scipy.io.loadmat(timestamps_path)
            frame_num_to_timestamp = frame_num_to_timestamp["unixDateTime"]  # TODO these are floats, a big no

            rosDS_path = scene_dir + '/rosDS'
            lidar_path = rosDS_path + '/rslidar_points_clean'
            camera_dir = rosDS_path + '/ueye_left_image_rect_color'

            if params['cfar_folder'] is not None:
                cfar_dir = rosDS_path + '/' + params['cfar_folder']

            # get lidar timestamps
            lidar_timestamps_and_paths = data_preparation.get_timestamps_and_paths(lidar_path)
            camera_timestamps_and_paths = data_preparation.get_timestamps_and_paths(camera_dir)
            if params['cfar_folder'] is not None:
                cfar_timestamps_and_paths = data_preparation.get_timestamps_and_paths(cfar_dir)

            # Get GPS
            #gps_topic= TopicIndex('gps_odom', rosDS_path)
            #gps_topic.load()

            for index in indices:
                aux_data_dict[aux_array_index] = {}

                ## handle radar
                aux_data_dict[aux_array_index]["elevation_path"] = os.path.join(cubes_dir,
                                                                                "Ele_Frame_" + str(index) + ".mat")
                aux_data_dict[aux_array_index]["power_path"] = os.path.join(cubes_dir,
                                                                            "Pow_Frame_" + str(index) + ".mat")
                aux_data_dict[aux_array_index]["dop_fold_path"] = os.path.join(cubes_dir,
                                                                                "DopFold_Frame_" + str(index) + ".mat")

                aux_data_dict[aux_array_index]["timestamp"] = (frame_num_to_timestamp[index - 1][0]) * 10 ** 9
                aux_data_dict[aux_array_index]["numpy_cube_path"] = os.path.join(cubes_dir,
                                                                                 "radar_cube_" + str(
                                                                                     index) + ".npy")
                ## handle LiDAR
                closest_lidar_time = data_preparation.closest_timestamp(aux_data_dict[aux_array_index]["timestamp"],
                                                                        lidar_timestamps_and_paths)
                aux_data_dict[aux_array_index]["gt_path"] = lidar_timestamps_and_paths[closest_lidar_time]
                aux_data_dict[aux_array_index]["gt_timestamp"] = closest_lidar_time

                ## handle camera
                closest_cam_time = data_preparation.closest_timestamp(aux_data_dict[aux_array_index]["timestamp"],
                                                                      camera_timestamps_and_paths)
                aux_data_dict[aux_array_index]["cam_path"] = camera_timestamps_and_paths[closest_cam_time]
                aux_data_dict[aux_array_index]["cam_timestamp"] = closest_cam_time

                ## handle GPS
                #aux_data_dict[aux_array_index]["gps"] = gps_topic.get_temporally_closest_message(aux_data_dict[aux_array_index]["timestamp"]).data


                ## handle CFAR
                if params['cfar_folder'] is not None:
                    closest_cfar_time = data_preparation.closest_timestamp(aux_data_dict[aux_array_index]["timestamp"],
                                                                           cfar_timestamps_and_paths)
                    aux_data_dict[aux_array_index]["cfar_path"] = cfar_timestamps_and_paths[closest_cfar_time]
                    aux_data_dict[aux_array_index]["cfar_timestamp"] = closest_cfar_time

                aux_array_index = aux_array_index + 1

        keys = list(aux_data_dict.keys())
        for i in range(0, len(keys), window_size):
            group = {keys[j]: aux_data_dict[keys[j]] for j in range(i, min(i + 3, len(keys)))}
            self.data_dict[global_array_index] = group
            global_array_index = global_array_index + 1

        # print division line
        print("--------------------------------------------------")

        print(mode + " dataset loaded with " + str(len(self.data_dict)) + " samples")
        print("scenes used: " + str(scene_set))
        # print division line
        print("--------------------------------------------------")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):

        # load elevation and power
        # elevation = scipy.io.loadmat(self.data_dict[idx]["elevation_path"])["elevationIndex"]

        # elevation = elevation.astype(np.single)
        # print(np.sum(np.isnan(elevation)))
        # elevation = np.nan_to_num(elevation, nan=22.0)
        # elevation = (elevation - np.mean(elevation)) / np.std(elevation)

        input_cube_full = np.array([])
        gt_cube_full = np.array([])
        item_params_full = []
        for index in self.data_dict[idx].keys():
            if not self.params['bev']:
                elevation = scipy.io.loadmat(self.data_dict[idx][index]["elevation_path"])["elevationIndex"]
                elevation = elevation.astype(np.single)
                elevation = np.nan_to_num(elevation, nan=17.0)
                elevation = elevation / 34

            power = scipy.io.loadmat(self.data_dict[idx][index]["power_path"])["radarCube"]

            power = power.astype(np.single)
            # Hardcoded maximum value after data exploration
            power = power / 8998.5576

            # Set 90% of the cube=0
            if self.params['quantile']:
                power[np.where(power < np.quantile(power, 0.9))] = 0

            # combine them into a single cube with 2 channels if not BEV
            if not self.params['bev']:
                input_cube = np.stack((power, elevation))
            else:
                input_cube = power
            # load gt
            gt_cloud = data_preparation.read_pointcloud(self.data_dict[idx][index]["gt_path"], mode="rs_lidar_clean")

            item_params = self.data_dict[idx][index]  # this is a dictionary with all the paths and timestamps
            gt_cube = data_preparation.lidarpc_to_lidarcube(gt_cloud, self.params)

            if not self.params['bev']:
                zero_pad = np.zeros([2, 12, 128, 240], dtype='single')
                input_cube = np.concatenate([input_cube, zero_pad], axis=1)
                zero_pad = np.zeros([2, 512, 128, 8], dtype='single')
                input_cube = np.concatenate([zero_pad, input_cube, zero_pad], axis=3)
                input_cube = np.transpose(input_cube, (0, 2, 1, 3))  # (C, H, W)

            else:
                zero_pad = np.zeros([12, 128, 240], dtype='single')
                input_cube = np.concatenate([input_cube, zero_pad])
                zero_pad = np.zeros([512, 128, 8], dtype='single')
                input_cube = np.concatenate([zero_pad, input_cube, zero_pad], 2)
                input_cube = np.transpose(input_cube, (1, 0, 2))  # (C, H, W)

            input_cube = np.expand_dims(input_cube,axis=0)
            input_cube_full = np.concatenate((input_cube_full, input_cube),
                                             axis=0) if input_cube_full.size else input_cube

            gt_cube = np.expand_dims(gt_cube, axis=0)
            gt_cube_full = np.concatenate((gt_cube_full, gt_cube),
                                             axis=0) if gt_cube_full.size else gt_cube

            item_params_full.append(item_params)

        return input_cube_full, gt_cube_full, item_params_full
