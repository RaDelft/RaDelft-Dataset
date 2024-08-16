import numpy as np
import os
from numpy.lib.recfunctions import structured_to_unstructured
import random
import scipy.io
import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F

# How to install Patchwork++
# https://github.com/url-kaist/patchwork-plusplus
#import pypatchworkpp

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def convert_pointcloud_from_mat_to_npy(input_dir):
    """
    Converts a point cloud from a .mat into a numpy array.
    It has to be called if the point cloud from the CFAR is run in MATLAB.
    It deletes the .mat file and replace it with the .npy files.

    :param input_dir: directory containing .mat files
    """
    all_files = os.listdir(input_dir)
    for file in all_files:
        if file.endswith(".mat"):
            load_path = os.path.join(input_dir, file)
            point_cloud = scipy.io.loadmat(load_path)['points']
            save_path = load_path.replace('mat', 'npy')
            np.save(save_path, point_cloud)
            os.remove(load_path)


def clean_and_save_lidar(input_dir):
    """
    Takes the lidar point cloud, crop it to the FoV of the radar and remove the ground reflections.
    It assumes the lidar points are in rslidar_points as in the provided datset.
    It saved them in the directory rslidar_points_clean.

    :param input_dir: Dataset Directory
    """

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


def non_uniform_voxelize_numpy(point_cloud, x_axis, y_axis, z_axis):
    """
    Voxelise a pointcloud in a non-uniform cube. The non-uniform axis are given as input to the function.
    In practise is used to voxelise the lidar point cloud into the radar non-uniform cube.

    :param point_cloud: the point cloud to voxelise
    :param x_axis: the first axis of the non-uniform cube
    :param y_axis: the second axis of the non-uniform cube
    :param z_axis: the third axis of the non-uniform cube
    :return: the cube with ones where there are at least one point and zero otherwise
    """
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
    """
    Generic function to load a point cloud.

    :param pointcloud_file: point cloud file
    :param mode: 'lidar' or 'radar' or 'rs_lidar' or 'rs_lidar_clean', depend on the format to be loaded
    :return: the point cloud in a numpy array
    """
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



def cleaning_ego_car(pointcloud):
    """
    Generic function to remove the close points

    :param pointcloud: point cloud in numpy format
    :return: the clean pointcloud
     """

    # remove points that are in the ego car, i.e. closer than 2 meters along x, and closer than 1 meter along y in both directions
    pointcloud = pointcloud[~((pointcloud[:, 0] < 1) & (pointcloud[:, 1] < 1) & (pointcloud[:, 1] > -1))]
    return pointcloud


def remove_ground_points_patchwork(point_cloud):
    """
    Function to remove all the ground reflections using the patchwork++ algorithm.
    In practise is used to clean the lidar data.

    :param point_cloud: the point cloud in numpy format
    :return: the clean pointcloud
    """

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


    return outliers, inliers


def closest_timestamp(new_timestamp, timestamps_dict):
    """
    It finds the closest timestamp in a dictionary to the gicen timestamp. Used to generate the dataset.
    For example, it can be used to find the closes camera frame to a radar frame.

    :param new_timestamp: the timestamp that you want to match
    :param timestamps_dict: a dictionary containing many timestamps from another sensor.
    :return: the closest timstamp
    """
    # Find the closest timestamp from a dictionary of timestamps.
    closest_time = min(timestamps_dict.keys(), key=lambda t: abs(t - new_timestamp))
    return closest_time


def get_timestamps_and_paths(directory):
    """
    It generates a dictionary with paths and timstamps from a given directory.
    It is used to generate the dictionaries for training the neural network.
    It assumes the timestamp is in the name of the file in ROS format.

    :param directory: the directory to find the files
    :return: the dictionary with timestamps and file names
    """

    # Gets a dictionary with timestamps as keys and file paths as values from a directory.
    timestamps_paths = {}

    for filename in os.listdir(directory):
        if filename.endswith(".npy") or filename.endswith(".jpg") or filename.endswith(".mat"):
            seconds, nanoseconds = filename.split('.')[0:2]
            # Convert seconds and nanoseconds to a single integer timestamp (in nanoseconds)
            timestamp = int(seconds) * 10 ** 9 + int(nanoseconds)
            file_path = os.path.join(directory, filename)
            timestamps_paths[timestamp] = file_path

    return timestamps_paths

def transform_point_cloud(point_cloud, rotation_angles, translation):
    """
    Transform a 3D point cloud by rotating and translating it.

    :param point_cloud (numpy.ndarray): 3D point cloud as a NumPy array with shape (N, 3),
            where N is the number of points.
    :param rotation_angles (tuple or list): Angles for rotation around the X, Y, and Z axes
            in degrees. For example, (45, 30, 60) for 45 degrees around X, 30 degrees
            around Y, and 60 degrees around Z.
    :param translation (tuple or list): Translation values along the X, Y, and Z axes.
            For example, (1.0, 2.0, 3.0) for translation of (1.0, 2.0, 3.0).

    :return: numpy.ndarray: Transformed point cloud as a NumPy array.
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
    """
    Loss used to train the NN in the non-time version.
    It removes the padding of the cube, reshape it and call the FocalLoss.

    :param radarcube: the radar cube
    :param lidarcube: the lidar cube
    :param params: the parameters, if not given, it will be initialized with the default parameters
    :return: the loss
    """

    if params is None:
        params = get_default_params()  # get the default parameters if no parameters are given

    if not params['bev']:
        batch_size = lidarcube.shape[0]
        radarcube = radarcube[:, :, :-12, 8:-8]
        radarcube = radarcube.contiguous()

        lidarcube = lidarcube.view(batch_size, 500, -1)
        radarcube = radarcube.view(batch_size, 1, 500, -1)
        metric = smp.losses.FocalLoss('binary', alpha=0.995, gamma=2)
    else:
        radarcube = radarcube[:, :, :-12, 8:-8]
        radarcube = radarcube.contiguous()
        metric = smp.losses.FocalLoss('binary', alpha=0.95, gamma=2)

    return metric(radarcube, lidarcube)


def radarcube_lidarcube_loss_time(radarcube, lidarcube, params):
    """
        Loss used to train the NN in the time version.
        It removes the padding of the cube, reshape it and call the FocalLoss.

        :param radarcube: the radar cube
        :param lidarcube: the lidar cube
        :param params: the parameters, if not given, it will be initialized with the default parameters
        :return: the loss
    """
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
    """
    It converts the lidar point cloud into a lidar cube using the same axis as the radar cube.
    The axis are not uniform.

    :param lidar_pc: the lidar point cloud
    :param params: the parameters, if not given, it will be initialized with the default parameters
    :return: the cube with ones where there are at least one point and zero otherwise
    """
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


def cube_to_pointcloud(cube, params, radar_cube, mode='radar', dop_fold_path=None):
    """
    It finds those voxels that are 1 and transform the to Cartesian point cloud.

    :param cube: the cube to be converted
    :param params: the parameters, if not given, it will be initialized with the default parameters
    :param radar_cube: the original radar cube to get the power from. Only used if mode == 'radar'
    :param mode: the origin of the cube to be converted. It can be lidar or radar
    :param dop_fold_path: path to the dop fold file. It is used to compute the velocity after the extension
                          of the ambiguity. If none, the ambiguous speed is returned.
    :return: the point cloud in cartesian coordinates.
    """
    if params is None:
        params = get_default_params()  # get the default parameters if no parameters are given

    # Get the axis
    range_axis = params['range_axis']
    azimuth_axis = params['azimuth_axis']
    elevation_axis = params['elevation_axis']
    vel_fft_size = params['vel_fft_size']
    vel_bin_size = params['vel_bin_size']

    if mode == 'radar':
        cube = torch.squeeze(cube)

        # Reshape to original size
        if params['bev']:
            cube = cube[:-12, 8:-8]
        else:
            cube = cube[:, :-12, 8:-8]

        # Apply sigmoid and find non-zero
        cube = F.sigmoid(cube)
        cube[cube < 0.55] = 0
        nonzero_indices = torch.nonzero(cube)
        nonzero_indices = nonzero_indices.numpy()

        # Convert to range, azimuth and elevation
        if params['bev']:
            range_indices = nonzero_indices[:, 0]
            azimuth_indices = nonzero_indices[:, 1]
            elevation_values = np.zeros(np.size(range_indices))
            # Select the power cube
        else:
            range_indices = nonzero_indices[:, 1]
            azimuth_indices = nonzero_indices[:, 2]
            elevation_values = elevation_axis[nonzero_indices[:, 0] - 1]
            # Select the power cube
            radar_cube = radar_cube[0, :, :, :]

        range_values = range_axis[range_indices]
        azimuth_values = azimuth_axis[azimuth_indices]
        # Get the Doppler info
        # Find the max Doppler
        doppler_indices = np.argmax(radar_cube[:, range_indices, azimuth_indices], 0)
        # Convert to speed
        vel_value = (doppler_indices - vel_fft_size / 2) * vel_bin_size

        # If the unfolded Doppler is provided, correct the speed value
        if dop_fold_path is not None:
            doppler_fold = scipy.io.loadmat(dop_fold_path)["dopplerFold"]
            doppler_fold = doppler_fold - 1  # Matlab Python 1 to 0
            doppler_corrected = doppler_fold[nonzero_indices[:, 1], doppler_indices]
            vel_value = (doppler_corrected - vel_fft_size / 2) * vel_bin_size

    # If the cube is a lidar cube
    else:
        cube = torch.squeeze(cube)
        nonzero_indices = torch.nonzero(cube)
        nonzero_indices = nonzero_indices.numpy()
        range_values = range_axis[nonzero_indices[:, 1]]
        azimuth_values = azimuth_axis[nonzero_indices[:, 2]]
        elevation_values = elevation_axis[nonzero_indices[:, 0] - 1]

    # Cartesian conversion
    radar_pc = spherical_to_cartesian(range_values, azimuth_values, elevation_values)

    # Only radar has velocity values
    if mode == 'radar':
        vel_value = np.expand_dims(vel_value, 1)
        radar_pc = np.hstack((radar_pc, vel_value))

    return radar_pc


def prepare_lidar_pointcloud(lidar_point_cloud, params):
    """
    Remove ground reflections, rotate to the radar frame, crop the FoV to fit the radar.

    :param lidar_point_cloud: the lidar point cloud
    :param params: the parameters, if not given, it will be initialized with the default parameters
    :return: the clean point cloud.
    """
    if params is None:
        params = get_default_params()  # get the default parameters if no parameters are given

    x_offset = params['x_offset']
    y_offset = params['y_offset']
    azimuth_offset = params['azimuth_offset']


    # Remove ground points
    lidar_point_cloud, _ = remove_ground_points_patchwork(lidar_point_cloud)
    lidar_point_cloud = lidar_point_cloud[:, 0:3]
    lidar_point_cloud = lidar_point_cloud[lidar_point_cloud[:, 2] > -2]

    # Rotate and Translate
    lidar_point_cloud = transform_point_cloud(lidar_point_cloud, [0, 0, azimuth_offset],
                                              [x_offset / 100, y_offset / 100, 0])

    # Filter point in the FoV of the radar
    lidar_point_cloud = filter_point_cloud_spherical(lidar_point_cloud, None)

    lidar_point_cloud = cleaning_ego_car(lidar_point_cloud)

    return lidar_point_cloud


def filter_point_cloud_spherical(point_cloud, params):
    """
    FoV crop to math the radar field of view. In spherical.

    :param point_cloud: the lidar point cloud
    :param params: the parameters, if not given, it will be initialized with the default parameters
    :return: the cropped point cloud.
    """
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
    """
    Generic cartesian to spherical.

    :param x: the first coordinate of the point
    :param y: the second coordinate of the point
    :param z: the third coordinate of the point
    :return: the range,az,el point in radians
    """
    range_ = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))

    return np.stack([range_, azimuth, elevation], axis=1)


def spherical_to_cartesian(range_, azimuth, elevation):
    """
    Generic spherical to cartesian.

    :param range_: the range of the point.
    :param azimuth: the azimuth in radians
    :param elevation: the elevation in radians
    :return: the cartesian point.
    """
    x = range_ * np.cos(elevation) * np.cos(azimuth)
    y = range_ * np.cos(elevation) * np.sin(azimuth)
    z = range_ * np.sin(elevation)
    return np.stack([x, y, z], axis=1)


def get_default_params():
    """
    Generates the default parameters for the processing.
    The ROS_DS_PATH is the only one that needs to be changed.
    :return: the parameters
    """

    ### some predefined parameters ###
    voxel_size = np.asarray([0.2, 0.2, 0.2])
    grid_range = np.asarray([0, -40, -3, 50, +40, 4])  # size of the grid in meters
    roi_spherical = np.asarray([0, -70, -15, 50, 70, 15])  # range azimuth elevation [min max]
    x_offset = 0
    y_offset = 0
    azimuth_offset = 7

    grid_size = np.floor((grid_range[3:] - grid_range[:3]) / voxel_size).astype(
        np.int32)  # calculate the size of the grid/pixels

    # Path to the dataset
    dataset_path = 'PATH_TO_DATASET'

    # Radar Cube axes
    # Range Axis
    range_cell_size = 0.1004
    max_range = 51.4242
    range_axis = np.arange(range_cell_size, max_range + range_cell_size, range_cell_size)
    range_axis = range_axis[10:-3]

    # Azimuth Axis
    angle_fft_size = 256
    wx_vec = np.linspace(-np.pi, np.pi, angle_fft_size)
    wx_vec = wx_vec[8:248]
    azimuth_axis = np.arcsin(wx_vec / (2 * np.pi * 0.4972))

    # Elevation Axis
    ele_fft_size = 128
    wz_vec = np.linspace(-np.pi, np.pi, ele_fft_size)
    wz_vec = wz_vec[47:81]
    elevation_axis = np.arcsin(wz_vec / (2 * np.pi * 0.4972))

    # Vel Axis
    vel_fft_size = 128
    vel_bin_size = 0.04607058455831936
    vel_fold = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

    # combine the parameters into a dictionary
    params = {'voxel_size': voxel_size,
              'grid_range': grid_range,
              'grid_size': grid_size,
              'dataset_path': dataset_path,
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
              'bev': False,
              'cfar_folder': None,
              'quantile': False,
              }

    return params


# main function
if __name__ == '__main__':

    # Convert CFAR mat to npy
    #path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment2/rosDS/radar_ososos/'
    #convert_pointcloud_from_mat_to_npy(path)
    #path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment6/rosDS/radar_ososos/'
    #convert_pointcloud_from_mat_to_npy(path)
    #path = '/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment2/rosDS/radar_ososos2D/'
    #convert_pointcloud_from_mat_to_npy(path)


    '''
    # Clean Lidar Points
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
