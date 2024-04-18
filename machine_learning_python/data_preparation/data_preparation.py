import numpy as np
from typing import Tuple
import matplotlib as mlp
import matplotlib.pyplot as plt
import os

from numpy.lib.recfunctions import structured_to_unstructured

from sklearn.linear_model import RANSACRegressor
import random
import scipy.io
import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import pypatchworkpp
from scipy.spatial.distance import directed_hausdorff

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def convert_cubes_from_mat_to_numpy(input_dir):
    all_files = os.listdir(input_dir)
    elevation_files = [file for file in all_files if "Ele_Frame" in file]
    power_files = [file for file in all_files if "Pow_Frame" in file]

    # get the list of numbers from the filenames bor both
    elevation_numbers = [int(file.split("_")[-1].split(".")[0]) for file in elevation_files]
    power_numbers = [int(file.split("_")[-1].split(".")[0]) for file in power_files]

    indices = elevation_numbers.copy()

    # make a dictionary, indices are keys, elevation, power, and gt_paths are values
    data_dict = {}
    for index in indices:
        data_dict[index] = {}

        ## handle radar
        data_dict[index]["elevation_path"] = os.path.join(input_dir, "Ele_Frame_" + str(index) + ".mat")
        data_dict[index]["power_path"] = os.path.join(input_dir, "Pow_Frame_" + str(index) + ".mat")

    # iterate over the dictionary, load both elevation and power, and save them as numpy arrays combined, in the same folder

    for index in indices:
        # progress bar
        if os.path.exists(os.path.join(input_dir, "radar_cube_" + str(index) + ".npy")):
            continue

        idx = index
        print("processing index: {}".format(index))

        elevation = scipy.io.loadmat(data_dict[idx]["elevation_path"])["elevationIndex"]
        power = scipy.io.loadmat(data_dict[idx]["power_path"])["radarCube"]

        # ToDo: Check this. Does it make sense to do it here?
        elevation = elevation.astype(np.single)
        power = power.astype(np.single)
        elevation = np.nan_to_num(elevation, nan=22.0)
        elevation = elevation / 44.0
        power = power / 1000.0
        # combine them into a single cube with 2 channels
        input_cube = np.stack((elevation, power), axis=3)
        # save the cube
        np.save(os.path.join(input_dir, "radar_cube_" + str(index) + ".npy"), input_cube)


def convert_pointcloud_from_mat_to_npy(input_dir):
    all_files = os.listdir(input_dir)
    for file in all_files:
        load_path = os.path.join(input_dir, file)
        point_cloud = scipy.io.loadmat(load_path)['points']
        save_path = load_path.replace('mat', 'npy')
        np.save(save_path, point_cloud)
        os.remove(load_path)


def clean_and_save_lidar(input_dir):
    params = get_default_params()

    rs_lidar_path = os.path.join(input_dir, 'rslidar_points')
    all_files = os.listdir(rs_lidar_path)
    save_dir = os.path.join(input_dir, 'rslidar_points_clean')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file in all_files:
        read_path = os.path.join(input_dir, 'rslidar_points', file)
        save_path = os.path.join(save_dir, file)
        gt_cloud = read_pointcloud(read_path, mode="rs_lidar")

        #   gt_cloud = gt_cloud[:, 0:3]
        gt_cloud = prepare_lidar_pointcloud(gt_cloud, None)

        np.save(save_path, gt_cloud)


def voxelize(points, params=None):
    """
    Converts 3D point cloud to a sparse voxel grid
    """

    if params is None:
        params = get_default_params()  # get the default parameters if no parameters are given

    voxel_size = params['voxel_size']
    grid_range = params['grid_range']
    max_points_in_voxel = params['max_points_in_voxel']
    max_num_voxels = params['max_num_voxels']

    points_copy = points.copy()  # copy points
    grid_size = np.floor((grid_range[3:] - grid_range[:3]) / voxel_size).astype(np.int32)  # calculate grid size

    coor_to_voxelidx = np.full((grid_size[2], grid_size[1], grid_size[0]), -1,
                               dtype=np.int32)  # create empty array with -1
    voxels = np.zeros((max_num_voxels, max_points_in_voxel, points.shape[-1]),
                      dtype=points_copy.dtype)  # create empty array for voxels
    coordinates = np.zeros((max_num_voxels, 3), dtype=np.int32)  # create empty array for coordinates for max_num_voxel
    num_points_per_voxel = np.zeros(max_num_voxels,
                                    dtype=np.int32)  # create empty array for num_points_per_voxel for max_num_voxel

    points_coords = np.floor((points_copy[:, :3] - grid_range[:3]) / voxel_size).astype(
        np.int32)  # calculate relative coordinates for points
    mask = ((points_coords >= 0) & (points_coords < grid_size)).all(1)  # create mask for points that are in the grid
    points_coords = points_coords[mask, ::-1]  # reverse coordinates
    points_copy = points_copy[mask]
    assert points_copy.shape[0] == points_coords.shape[0]

    voxel_num = 0
    for i, coord in enumerate(points_coords):
        voxel_idx = coor_to_voxelidx[tuple(coord)]
        if voxel_idx == -1:
            voxel_idx = voxel_num
            voxel_num += 1
            coor_to_voxelidx[tuple(coord)] = voxel_idx
            coordinates[voxel_idx] = coord
        point_idx = num_points_per_voxel[voxel_idx]
        if point_idx < max_points_in_voxel:
            voxels[voxel_idx, point_idx] = points_copy[i]
            num_points_per_voxel[voxel_idx] += 1

    return voxels[:voxel_num], coordinates[:voxel_num], num_points_per_voxel[:voxel_num]


def non_uniform_voxelize(point_cloud, x_axis, y_axis, z_axis):
    # Calculate the number of voxels along each axis
    num_x = len(x_axis)
    num_y = len(y_axis)
    num_z = len(z_axis)

    # Initialize the voxel grid with zeros as boolean tensors
    voxel_grid = torch.zeros((num_x, num_y, num_z), device='cuda:0')

    # Expand point_cloud to have the same shape as voxel_grid
    point_cloud = point_cloud.unsqueeze(1)  # Add a dimension for broadcasting

    # Calculate voxel indices for each axis using broadcasting
    x_indices = torch.searchsorted(x_axis, point_cloud[..., 0].contiguous(), right=True)
    y_indices = torch.searchsorted(y_axis, point_cloud[..., 1].contiguous(), right=True)
    z_indices = torch.searchsorted(z_axis, point_cloud[..., 2].contiguous(), right=True)

    # Create a mask for valid indices
    valid_indices = (x_indices >= 0) & (x_indices < num_x) & (y_indices >= 0) & (y_indices < num_y) & (
            z_indices >= 0) & (z_indices < num_z)

    # Mark the voxel as occupied using the mask
    voxel_grid[x_indices[valid_indices], y_indices[valid_indices], z_indices[valid_indices]] = 1

    return voxel_grid


def non_uniform_voxelize_numpy(point_cloud, x_axis, y_axis, z_axis):
    num_x = len(x_axis)
    num_y = len(y_axis)
    num_z = len(z_axis)

    # Initialize the voxel grid with zeros as boolean tensors
    voxel_grid = np.zeros((num_x, num_y, num_z))

    # Calculate voxel indices for each axis using broadcasting
    x_indices = np.searchsorted(x_axis, point_cloud[..., 0], side='left')
    y_indices = np.searchsorted(y_axis, point_cloud[..., 1], side='left')
    z_indices = np.searchsorted(z_axis, point_cloud[..., 2], side='left')

    # This is just in case there are some points outside the grid. It should not be the case since we clean first
    # the lidar point cloud.
    valid_indices = (x_indices >= 0) & (x_indices < num_x) & (y_indices >= 0) & (y_indices < num_y) & (
            z_indices >= 0) & (z_indices < num_z)
    x_indices = x_indices[valid_indices]
    y_indices = y_indices[valid_indices]
    z_indices = z_indices[valid_indices]
    point_cloud = point_cloud[valid_indices, :]

    # Correct the indices, so they are the closest, and not always the left ones.
    condition = (x_indices > 0) & ((x_indices == num_x) | (
                np.abs(point_cloud[..., 0] - x_axis[x_indices - 1]) < np.abs(point_cloud[..., 0] - x_axis[x_indices])))

    x_indices[condition] = x_indices[condition] - 1

    condition = (y_indices > 0) & ((y_indices == num_y) | (
            np.abs(point_cloud[..., 1] - y_axis[y_indices - 1]) < np.abs(point_cloud[..., 1] - y_axis[y_indices])))

    y_indices[condition] = y_indices[condition] - 1

    condition = (z_indices > 0) & ((z_indices == num_z) | (
            np.abs(point_cloud[..., 2] - z_axis[z_indices - 1]) < np.abs(point_cloud[..., 2] - z_axis[z_indices])))

    z_indices[condition] = z_indices[condition] - 1

    # Mark the voxel as occupied using the mask
    voxel_grid[x_indices, y_indices, z_indices] = 1

    return voxel_grid


def read_pointcloud(pointcloud_file, mode='lidar'):
    pointcloud = np.load(pointcloud_file)  # read pointcloud

    if mode == 'lidar':  # Velodyne has x, y, z, intensity, ring index, and timestamp
        pointcloud = structured_to_unstructured(pointcloud)
        pointcloud = pointcloud.reshape((-1, 6))

    elif mode == 'radar':
        pointcloud = pointcloud[:, 0:3]
        #pointcloud = structured_to_unstructured(pointcloud)
    # pointcloud = pointcloud.reshape((-1, 7))

    elif mode == 'rs_lidar':
        pointcloud = structured_to_unstructured(pointcloud)
        pointcloud = pointcloud.reshape((-1, 4))

    elif mode == 'rs_lidar_clean':
        pointcloud = pointcloud.reshape((-1, 3))

    return pointcloud


# convert the voxels to a point cloud
def voxels_to_metric_coord(voxels, params):
    if params is None:
        params = get_default_params()
    voxel_size = params['voxel_size']
    grid_range = params['grid_range']

    x, y, z = voxels

    # convert x y to meters, i.e. multiply by voxel size and add grid range
    x = x * voxel_size[0] + grid_range[0]
    y = y * voxel_size[1] + grid_range[1]
    z = z * voxel_size[2] + grid_range[2]

    # all has to be shifted by half voxel size
    x += voxel_size[0] / 2
    y += voxel_size[1] / 2
    z += voxel_size[2] / 2

    return x, y, z


def crop_pointcloud_to_gridrange(pointcloud, params=None):
    if params is None:
        params = get_default_params()

    grid_range = params['grid_range']
    # get the indices of the points that are within the grid range
    idx = np.where((pointcloud[:, 0] >= grid_range[0]) & (pointcloud[:, 0] <= grid_range[3]) & (
            pointcloud[:, 1] >= grid_range[1]) & (pointcloud[:, 1] <= grid_range[4]))
    # get points that are within the grid range
    pointcloud = pointcloud[idx]

    return pointcloud


def cleaning_ego_car(pointcloud):
    # remove points that are in the ego car, i.e. closer than 2 meters along x, and closer than 1 meter along y in both directions
    pointcloud = pointcloud[~((pointcloud[:, 0] < 1) & (pointcloud[:, 1] < 1) & (pointcloud[:, 1] > -1))]
    return pointcloud


def remove_ground_points(point_cloud, distance_threshold=0.3, method='svd', only_road=False):
    if only_road == True and method is not "ransac_sklearn":
        raise ValueError("Only road is only supported for ransac_sklearn method")

    if method == 'svd':
        # Fit a plane to the point cloud using a singular value decomposition (SVD) approach
        mean = np.mean(point_cloud, axis=0)
        centered_cloud = point_cloud - mean
        U, s, V = np.linalg.svd(centered_cloud, full_matrices=False)
        normal = V[-1, :]

        # Project each point onto the plane
        dot_products = np.dot(centered_cloud, normal)
        distances = np.abs(dot_products)


    elif method == 'ransac':
        # basic_cut = -0.3
        # original_point_cloud = point_cloud.copy()
        # point_cloud = point_cloud[point_cloud[:, 2] < basic_cut]

        # Fit a plane to the point cloud using the RANSAC algorithm
        best_model = None
        best_score = 0
        max_iterations = 2000

        for i in range(max_iterations):
            # Randomly select 3 points from the point cloud
            sample_indices = random.sample(range(point_cloud.shape[0]), 3)
            sample = point_cloud[sample_indices, :]

            # Fit a plane to the 3 points
            v1 = sample[1, :] - sample[0, :]
            v2 = sample[2, :] - sample[0, :]
            normal = np.cross(v1, v2)
            normal /= np.linalg.norm(normal)
            d = -np.dot(normal, sample[0, :])

            # Evaluate the quality of the model by counting the number of inliers
            distances = np.abs(np.dot(point_cloud, normal) + d)
            inlier_count = np.count_nonzero(distances < distance_threshold)

            # Update the best model if the current model is better
            if inlier_count > best_score:
                best_model = (normal, d)
                best_score = inlier_count

        # Remove points that are close to the ground plane
        normal, d = best_model
        distances = np.abs(np.dot(point_cloud, normal) + d)

        # return the indices of the points that are not close to the ground plane (i.e. the outliers) and the inlier points
        # returh both inliner and outlier points

    elif method == 'ransac_sklearn':
        original_point_cloud = point_cloud.copy()

        basic_cut = -0.3
        original_point_cloud = original_point_cloud[original_point_cloud[:, 2] < basic_cut]
        if only_road == True:
            # Filter for points that are on the road
            road_mask = point_cloud[:, 4] == 8  # 8 is the class label for road
            point_cloud = point_cloud[road_mask, :]

        # Fit a plane to the point cloud using the RANSAC algorithm from scikit-learn
        model = RANSACRegressor(estimator=None, min_samples=3, residual_threshold=distance_threshold,
                                random_state=None)
        model.fit(point_cloud[:, :2], point_cloud[:, 2])

        # Remove points that are close to the ground plane
        distances = np.abs(model.predict(original_point_cloud[:, :2]) - original_point_cloud[:, 2])

    elif method == 'patchwork':
        params = pypatchworkpp.Parameters()

        PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

        # Estimate Ground
        PatchworkPLUSPLUS.estimateGround(point_cloud)

        # Get Ground and Nonground
        ground = PatchworkPLUSPLUS.getGround()
        nonground = PatchworkPLUSPLUS.getNonground()  # we are interested in this, it is just a numpy array of shape (N,3)
        time_taken = PatchworkPLUSPLUS.getTimeTaken()

        ground_idx = PatchworkPLUSPLUS.getGroundIndices()
        nonground_idx = PatchworkPLUSPLUS.getNongroundIndices()

        outliers = nonground
        inliers = ground

    else:
        raise ValueError('method not supported')

    if not method == 'patchwork':
        outliers = np.where(distances > distance_threshold)[0]
        inliers = np.where(distances <= distance_threshold)[0]

    return outliers, inliers


def closest_timestamp(new_timestamp, timestamps_dict):
    # Find the closest timestamp from a dictionary of timestamps.

    closest_time = min(timestamps_dict.keys(), key=lambda t: abs(t - new_timestamp))
    return closest_time


def get_timestamps_and_paths(directory):
    # Gets a dictionary with timestamps as keys and file paths as values from a directory.

    timestamps_paths = {}

    for filename in os.listdir(directory):
        if filename.endswith(".npy") or filename.endswith(".jpg"):
            seconds, nanoseconds = filename.split('.')[0:2]
            # Convert seconds and nanoseconds to a single integer timestamp (in nanoseconds)
            timestamp = int(seconds) * 10 ** 9 + int(nanoseconds)
            file_path = os.path.join(directory, filename)
            timestamps_paths[timestamp] = file_path

    return timestamps_paths


def rotate_pointcloud(pointcloud, angle):
    # rotate the pointcloud 90 degrees around the z axis
    rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pointcloud[:, :3] = np.dot(pointcloud[:, :3], rotation_matrix)

    return pointcloud


def get_default_params():
    ### some predefined parameters ###
    voxel_size = np.asarray([0.2, 0.2, 0.2])
    grid_range = np.asarray([0, -40, -3, 50, +40, 4])  # size of the grid in meters
    roi_spherical = np.asarray([0, -70, -20, 50, 70, 20])  # range azimuth elevation [min max]
    max_points_in_voxel = 35  # the maximum number of points in a voxel, does not really matter for radar
    max_num_voxels = 50000  # the maximum number of voxels
    x_offset = 0
    y_offset = 0
    azimuth_offset = 7

    grid_size = np.floor((grid_range[3:] - grid_range[:3]) / voxel_size).astype(
        np.int32)  # calculate the size of the grid/pixels

    ROS_DS_Path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment1/rosDS/'

    # Radar Cube axes
    # Range Axis
    range_cell_size = 0.1004
    max_range = 51.4242
    range_axis = np.arange(range_cell_size, max_range + range_cell_size, range_cell_size)
    range_axis = range_axis[10:-3]

    # Azimuth Axis
    angle_fft_size = 256
    wx_vec = np.linspace(-np.pi, np.pi, angle_fft_size)
    # wx_vec = np.flip(wx_vec)
    wx_vec = wx_vec[8:248]
    azimuth_axis = np.arcsin(wx_vec / (2 * np.pi * 0.4972))

    # Elevation Axis
    ele_fft_size = 128
    wz_vec = np.linspace(-np.pi, np.pi, ele_fft_size)
    # wz_vec = np.flip(wz_vec)
    wz_vec = wz_vec[42:86]
    elevation_axis = np.arcsin(wz_vec / (2 * np.pi * 0.4972))

    # Vel Axis
    vel_fft_size = 128
    vel_bin_size = 0.04607058455831936
    vel_fold = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    # azimuth_axis = np.flip(azimuth_axis)
    # elevation_axis = np.flip(elevation_axis)
    # azimuth_axis = azimuth_axis.copy()
    #  elevation_axis = elevation_axis.copy()

    # ToDo: This does not look optimised. Is there a better way?
    # range_axis = torch.from_numpy(range_axis).float().to(torch.device('cuda:0'))
    # azimuth_axis = torch.from_numpy(azimuth_axis).float().to(torch.device('cuda:0'))
    # elevation_axis = torch.from_numpy(elevation_axis).float().to(torch.device('cuda:0'))

    loss_type = 'knn'  # Either iou or knn

    # combine the parameters into a dictionary
    params = {'voxel_size': voxel_size,
              'grid_range': grid_range,
              'max_points_in_voxel': max_points_in_voxel,
              'max_num_voxels': max_num_voxels,
              'grid_size': grid_size,
              'ROS_DS_Path': ROS_DS_Path,
              'range_axis': range_axis,
              'azimuth_axis': azimuth_axis,
              'elevation_axis': elevation_axis,
              'vel_bin_size': vel_bin_size,
              'vel_fft_size': vel_fft_size,
              'vel_fold': vel_fold,
              'roi_spherical': roi_spherical,
              'x_offset': x_offset,
              'y_offset': y_offset,
              'azimuth_offset': azimuth_offset,
              'loss_no_point': 100,
              'fixed_pad_size': 45000,
              'loss_weight': 0.5,
              'loss_type': loss_type,
              'bev': False,
              'label_smoothing': True,
              'cfar_folder': None,
              'quantile': False,
              }

    return params


def get_closest_lidar_pointCloud(timestamp):
    params = get_default_params()

    lidar_path = params['ROS_DS_Path'] + "/rslidar_points/"

    # get the timestamps and paths for the lidar pointclouds
    lidar_timestamps_paths = get_timestamps_and_paths(lidar_path)

    closest_time = closest_timestamp(timestamp, lidar_timestamps_paths)

    closest_lidar_pcl_path = lidar_timestamps_paths[closest_time]

    # read the lidar pointcloud
    lidar_pointcloud = read_pointcloud(closest_lidar_pcl_path, mode='rs_lidar')

    return lidar_pointcloud


def transform_point_cloud(point_cloud, rotation_angles, translation):
    """
    Transform a 3D point cloud by rotating and translating it.

    Args:
        point_cloud (numpy.ndarray): 3D point cloud as a NumPy array with shape (N, 3),
            where N is the number of points.
        rotation_angles (tuple or list): Angles for rotation around the X, Y, and Z axes
            in degrees. For example, (45, 30, 60) for 45 degrees around X, 30 degrees
            around Y, and 60 degrees around Z.
        translation (tuple or list): Translation values along the X, Y, and Z axes.
            For example, (1.0, 2.0, 3.0) for translation of (1.0, 2.0, 3.0).

    Returns:
        numpy.ndarray: Transformed point cloud as a NumPy array.
    """
    # Extract rotation angles
    rx, ry, rz = map(np.radians, rotation_angles)

    # Create rotation matrices
    rot_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])

    rot_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, np.cos(ry)]])

    rot_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                      [np.sin(rz), np.cos(rz), 0],
                      [0, 0, 1]])

    # Apply rotations
    rotated_cloud = point_cloud[:, 0:3]
    rotated_cloud = rotated_cloud.dot(rot_x).dot(rot_y).dot(rot_z)

    # Apply translation
    translated_cloud = rotated_cloud + np.array(translation)
    if point_cloud.shape[1] > 3:
        translated_cloud = np.hstack((translated_cloud, np.expand_dims(point_cloud[:, 3], 1)))

    return translated_cloud


def radarcube_lidarcube_loss(radarcube, lidarcube, params):
    if params is None:
        params = get_default_params()  # get the default parameters if no parameters are given

    if not params['bev']:
        batch_size = lidarcube.shape[0]
        radarcube = radarcube[:, :, :-12, 8:-8]
        radarcube = radarcube.contiguous()

        if params['label_smoothing']:
            for i in range(batch_size):
                lidarcube[i, :, :, :] = gaussian_blur(lidarcube[i, :, :, :], 0.5)

        lidarcube = lidarcube.view(batch_size, 500, -1)
        radarcube = radarcube.view(batch_size, 1, 500, -1)
        metric = smp.losses.FocalLoss('binary', alpha=0.995, gamma=2)
    else:
        radarcube = radarcube[:, :, :-12, 8:-8]
        radarcube = radarcube.contiguous()
        metric = smp.losses.FocalLoss('binary', alpha=0.95, gamma=2)

    return metric(radarcube, lidarcube)


def radarcube_lidarcube_loss_time(radarcube, lidarcube, params):
    if params is None:
        params = get_default_params()  # get the default parameters if no parameters are given

    batch_size = lidarcube.shape[0]

    if not params['bev']:
        radarcube = radarcube[:, :, :, :-12, 8:-8]
    else:
        radarcube = radarcube[:, :, :-12, 8:-8]

    radarcube = radarcube.contiguous()

    lidarcube = lidarcube.view(batch_size, 500, -1)
    radarcube = radarcube.view(batch_size, 1, 500, -1)
    metric = smp.losses.FocalLoss('binary', alpha=0.95, gamma=2)

    return metric(radarcube, lidarcube)


def lidarpc_to_lidarcube(lidar_pc, params):
    if params is None:
        params = get_default_params()  # get the default parameters if no parameters are given

    spher = cartesian_to_spherical(lidar_pc[:, 0], lidar_pc[:, 1], lidar_pc[:, 2])

    lidar_cube = non_uniform_voxelize_numpy(spher, params['range_axis'], params['azimuth_axis'],
                                            params['elevation_axis'])

    lidar_cube = np.flip(lidar_cube, axis=1)
    lidar_cube = np.flip(lidar_cube, axis=2)

    if not params['bev']:
        lidar_cube = np.transpose(lidar_cube, [2, 0, 1])
    else:
        lidar_cube = np.max(lidar_cube, axis=2)

    data = lidar_cube.astype('single').copy()
    del lidar_cube

    return data


def cube_to_pointcloud(cube, params, radar_cube, elevation_path, mode='radar', noElevation=False, dop_fold_path=None):
    if params is None:
        params = get_default_params()  # get the default parameters if no parameters are given

    range_axis = params['range_axis']
    azimuth_axis = params['azimuth_axis']
    elevation_axis = params['elevation_axis']
    vel_fft_size = params['vel_fft_size']
    vel_bin_size = params['vel_bin_size']
    vel_fold = params['vel_fold']

    if mode == 'radar':
        if params['bev']:
            cube = torch.squeeze(cube)
            cube = cube[:-12, 8:-8]
            cube = F.sigmoid(cube)
            cube[cube < 0.5] = 0
            nonzero_indices = torch.nonzero(cube[:, :])
            nonzero_indices = nonzero_indices.numpy()

            radar_cube = torch.squeeze(radar_cube)
            radar_cube = radar_cube[:, :-12, 8:-8]

            _, doppler_indices = torch.max(radar_cube[:, nonzero_indices[:, 0], nonzero_indices[:, 1]], dim=0)
            elevation = scipy.io.loadmat(elevation_path)["elevationIndex"]
            elevation_indices = elevation[nonzero_indices[:, 0], doppler_indices, nonzero_indices[:, 1]]

            range_values = range_axis[nonzero_indices[:, 0]]
            azimuth_values = azimuth_axis[nonzero_indices[:, 1]]
            elevation_values = elevation_axis[elevation_indices - 1]

        else:
            cube = torch.squeeze(cube)
            cube = cube[:, :-12, 8:-8]
            cube = F.sigmoid(cube)
            cube[cube < 0.5] = 0
            nonzero_indices = torch.nonzero(cube)
            nonzero_indices = nonzero_indices.numpy()
            range_values = range_axis[nonzero_indices[:, 1]]
            azimuth_values = azimuth_axis[nonzero_indices[:, 2]]
            elevation_values = elevation_axis[nonzero_indices[:, 0] - 1]
            radar_cube = radar_cube[0, :, :, :]
            # Load power cube to find the max value in Doppler
            doppler_indices = np.argmax(radar_cube[:, nonzero_indices[:, 1], nonzero_indices[:, 2]], 0)

            if dop_fold_path is not None:
                doppler_fold = scipy.io.loadmat(dop_fold_path)["dopplerFold"]
                doppler_fold = doppler_fold - 1  # Matlab Python 1 to 0
                doppler_corrected = doppler_fold[nonzero_indices[:, 1], doppler_indices]
                vel_value = (doppler_corrected - vel_fft_size / 2) * vel_bin_size
            if noElevation:
                elevation_values = np.zeros(elevation_values.shape)

    else:
        cube = torch.squeeze(cube)
        nonzero_indices = torch.nonzero(cube)
        nonzero_indices = nonzero_indices.numpy()
        range_values = range_axis[nonzero_indices[:, 1]]
        azimuth_values = azimuth_axis[nonzero_indices[:, 2]]
        elevation_values = elevation_axis[nonzero_indices[:, 0] - 1]

    # azimuth_axis = np.flip(azimuth_axis)
    # elevation_axis = np.flip(elevation_axis)

    radar_pc = spherical_to_cartesian(range_values, azimuth_values, elevation_values)

    if dop_fold_path is not None:
        vel_value = np.expand_dims(vel_value, 1)
        radar_pc = np.hstack((radar_pc, vel_value))

    return radar_pc


def prepare_lidar_pointcloud(lidar_point_cloud, params):
    if params is None:
        params = get_default_params()  # get the default parameters if no parameters are given

    x_offset = params['x_offset']
    y_offset = params['y_offset']
    azimuth_offset = params['azimuth_offset']

    #traceOriginal = go.Scatter3d(x=lidar_point_cloud[:, 0], y=lidar_point_cloud[:, 1], z=lidar_point_cloud[:, 2],
    #                                   mode='markers', marker=dict(size=2,color='blue',opacity=0.5))

    # Remove ground points
    lidar_point_cloud, _ = remove_ground_points(lidar_point_cloud, method="patchwork")
    #lidar_point_cloud = lidar_point_cloud[:, 0:3]
    #lidar_point_cloud = lidar_point_cloud[lidar_point_cloud[:, 2] > -2]

    # Rotate and Translate
    #lidar_point_cloud = transform_point_cloud(lidar_point_cloud, [0, 0, azimuth_offset],
    #                                          [x_offset / 100, y_offset / 100, 0])

    # Filter point in the FoV of the radar
    #lidar_point_cloud = filter_point_cloud_spherical(lidar_point_cloud, None)

    #tracePatch = go.Scatter3d(x=lidar_point_cloud[:, 0], y=lidar_point_cloud[:, 1], z=lidar_point_cloud[:, 2],
    #                          mode='markers', marker_symbol='cross', marker=dict(size=2, color='red', opacity=0.5))
    #fig = go.Figure(data=[traceOriginal, tracePatch])

    #fig.update_layout(
    #    margin=dict(l=0, r=0, b=0, t=0),
    #    scene=dict(aspectmode="data")
    #)
    #fig.show()

    #lidar_point_cloud = cleaning_ego_car(lidar_point_cloud)

    return lidar_point_cloud


def filter_point_cloud_spherical(point_cloud, params):
    if params is None:
        params = get_default_params()  # get the default parameters if no parameters are given

    roi_spherical = params['roi_spherical']

    # Convert Cartesian to Spherical Coordinates
    sphe = cartesian_to_spherical(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])

    # Define boolean masks for filtering
    range_mask = (sphe[:, 0] >= roi_spherical[0]) & (sphe[:, 0] <= roi_spherical[3])
    azimuth_mask = (sphe[:, 1] >= (roi_spherical[1] * np.pi / 180)) & (
            sphe[:, 1] <= (roi_spherical[4] * np.pi / 180))
    elevation_mask = (sphe[:, 2] >= (roi_spherical[2] * np.pi / 180)) & (
            sphe[:, 2] <= (roi_spherical[5] * np.pi / 180))

    # Combine the masks using logical 'and' to get the final filter
    final_mask = range_mask & azimuth_mask & elevation_mask

    # Apply the filter to the point cloud
    filtered_point_cloud = point_cloud[final_mask]

    return filtered_point_cloud


def cartesian_to_spherical(x, y, z):
    range_ = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))

    return np.stack([range_, azimuth, elevation], axis=1)


def spherical_to_cartesian(range_, azimuth, elevation):
    x = range_ * np.cos(elevation) * np.cos(azimuth)
    y = range_ * np.cos(elevation) * np.sin(azimuth)
    z = range_ * np.sin(elevation)
    return np.stack([x, y, z], axis=1)


def gaussian_blur(cube, blur_sigma=2):
    k = make_gaussian_kernel(blur_sigma)

    # Separable 1D convolution
    vol_in = cube[None, None, ...]
    vol_in = vol_in.float()
    # k2d = torch.einsum('i,j->ij', k, k)
    # k2d = k2d / k2d.sum() # not necessary if kernel already sums to zero, check:
    # print(f'{k2d.sum()=}')
    k1d = k[None, None, :, None, None]
    for i in range(3):
        vol_in = vol_in.permute(0, 1, 4, 2, 3)
        vol_in = F.conv3d(vol_in, k1d, stride=1, padding=(len(k) // 2, 0, 0))
    vol_3d_sep = vol_in
    vol_3d_sep = torch.squeeze(vol_3d_sep)

    return vol_3d_sep


def make_gaussian_kernel(sigma):
    ks = int(sigma * 5)
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks).cuda()
    gauss = torch.exp((-(ts / sigma) ** 2 / 2)).cuda()
    kernel = gauss / gauss.sum()

    return kernel


# main function
if __name__ == '__main__':
    # Convert CFAR mat to npy
    #path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment2/rosDS/radar_ososos/'
    #convert_pointcloud_from_mat_to_npy(path)
    #path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment6/rosDS/radar_ososos/'
    #convert_pointcloud_from_mat_to_npy(path)
    #path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment2/rosDS/radar_ososos2D/'
    #convert_pointcloud_from_mat_to_npy(path)
    # Clean Lidar Points

    '''
    path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment1/rosDS/'
    clean_and_save_lidar(path)
    print("FINISH 1 clean")    
    path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment2/rosDS/'
    clean_and_save_lidar(path)
    print("FINISH 2 clean")
    path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment3/rosDS/'
    clean_and_save_lidar(path)
    print("FINISH 3 clean")
    path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment4/rosDS/'
    clean_and_save_lidar(path)
    print("FINISH 4 clean")
    path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment5/rosDS/'
    clean_and_save_lidar(path)
    print("FINISH 5 clean")
    path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment6/rosDS/'
    clean_and_save_lidar(path)
    print("FINISH 6 clean")
    path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment7/rosDS/'
    clean_and_save_lidar(path)
    '''
