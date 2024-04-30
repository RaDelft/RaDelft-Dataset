import numpy as np
import glob
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
from data_preparation import data_preparation
import torchvision.transforms as transforms
from loaders.rad_cube_loader import RADCUBE_DATASET
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import os


def create_transformation_matrix(roll, pitch, yaw, translation_vector):
    """
    Create a transformation matrix from roll, pitch, yaw angles (in degrees) and a translation vector.

    Args:
    roll (float): Roll angle in degrees.
    pitch (float): Pitch angle in degrees.
    yaw (float): Yaw angle in degrees.
    translation_vector (list): A list of three elements representing the translation vector.

    Returns:
    numpy.ndarray: A 4x4 transformation matrix.
    """
    # Convert angles from degrees to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # Create individual rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])

    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))

    # rotate around y axis again h degrees
    h = 1
    h_rad = np.radians(h)
    R_h = np.array([[np.cos(h_rad), 0, np.sin(h_rad)],
                    [0, 1, 0],
                    [-np.sin(h_rad), 0, np.cos(h_rad)]])

    R = np.dot(R_h, R)
    # Create the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


def get_sensor_transforms():
    calibration_file = "../utils/calib.txt"
    with open(calibration_file, "r") as f:
        lines = f.readlines()
        intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
        extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
        extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)

    camera_projection_matrix, T_camera_lidar = intrinsic, extrinsic

    return camera_projection_matrix, T_camera_lidar


def homogeneous_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
This function applies the homogenous transform using the dot product.
    :param points: Points to be transformed in a Nx4 numpy array.
    :param transform: 4x4 transformation matrix in a numpy array.
    :return: Transformed points of shape Nx4 in a numpy array.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"{transform.shape} must be 4x4!")
    if points.shape[1] != 4:
        raise ValueError(f"{points.shape[1]} must be Nx4!")
    return transform.dot(points.T).T


def project_3d_to_2d(points: np.ndarray, projection_matrix: np.ndarray):
    """
This function projects the input 3d ndarray to a 2d ndarray, given a projection matrix.
    :param points: Homogenous points to be projected.
    :param projection_matrix: 4x4 projection matrix.
    :return: 2d ndarray of the projected points.
    """
    if points.shape[-1] != 4:
        raise ValueError(f"{points.shape[-1]} must be 4!")

    uvw = projection_matrix.dot(points.T)
    uvw /= uvw[2]
    uvs = uvw[:2].T
    uvs = np.round(uvs).astype(int)

    return uvs


def project_pcl_to_image(point_cloud, t_camera_pcl, camera_projection_matrix, image_shape):
    """
A helper function which projects a point clouds specific to the dataset to the camera image frame.
    :param point_cloud: Point cloud to be projected.
    :param t_camera_pcl: Transformation from the pcl frame to the camera frame.
    :param camera_projection_matrix: The 4x4 camera projection matrix.
    :param image_shape: Size of the camera image.
    :return: Projected points, and the depth of each point.
    """
    point_homo = np.hstack((point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))))

    radar_points_camera_frame = homogeneous_transformation(point_homo,
                                                           transform=t_camera_pcl)

    point_depth = radar_points_camera_frame[:, 2]

    uvs = project_3d_to_2d(points=radar_points_camera_frame,
                           projection_matrix=camera_projection_matrix)

    filtered_idx = canvas_crop(points=uvs,
                               image_size=image_shape,
                               points_depth=point_depth)

    # uvs = uvs[filtered_idx]
    # point_depth = point_depth[filtered_idx]

    return uvs, point_depth, filtered_idx


def canvas_crop(points, image_size, points_depth=None):
    """
This function filters points that lie outside a given frame size.
    :param points: Input points to be filtered.
    :param image_size: Size of the frame.
    :param points_depth: Filters also depths smaller than 0.
    :return: Filtered points.
    """
    idx = points[:, 0] > 0
    idx = np.logical_and(idx, points[:, 0] < image_size[1])
    idx = np.logical_and(idx, points[:, 1] > 0)
    idx = np.logical_and(idx, points[:, 1] < image_size[0])
    if points_depth is not None:
        idx = np.logical_and(idx, points_depth > 0)

    return idx


# function that takes axis and 2d points and plots them
def plot_points_on_image(ax, cax, points_2d, image, point_depths=None, size=40, color_by='depth', alpha=0.1):
    ax.imshow(image)
    if color_by == "depth" and point_depths is not None:
        # print("color by depth")
        order = np.argsort(-point_depths)
        point_depths = point_depths[order]
        points_2d = points_2d[order, :]

        im = ax.scatter(points_2d[:, 0], points_2d[:, 1], c=point_depths, s=size, alpha=alpha)
        cb = ax.figure.colorbar(im, cax=cax)
        cb.solids.set(alpha=1)
        cb.set_label('Range (m)')
    elif color_by == "height":
        print("color by height")
        ax.scatter(points_2d[:, 0], points_2d[:, 1], c=points_2d[:, 1], s=size)

    elif color_by == "depth_scale":
        print("color by depth scale")
        ax.scatter(points_2d[:, 0], points_2d[:, 1], c=point_depths, s=3 / point_depths * point_depths)
        # scale dots based on depth

    ax.axis('off')
    return cb


# filter by distance
def filter_by_distance(point_depths, threshold):
    """
This function filters points that are further than a given distance.
    :param points: Input points to be filtered.
    :param distance: Distance threshold.
    :return: Filtered points.
    """
    return point_depths < threshold


def filter_point_cloud(point_cloud):
    # remove behind sensor:
    point_cloud = point_cloud[point_cloud[:, 0] > 0, :]

    # remove to far in x
    point_cloud = point_cloud[point_cloud[:, 0] < 30, :]

    # remove to far in y
    point_cloud = point_cloud[point_cloud[:, 1] < 10, :]
    point_cloud = point_cloud[point_cloud[:, 1] > -10, :]

    return point_cloud


def plot_bev(pointcloud, ax, cax, cblabel=True):
    """
This function plots the BEV of a point cloud.
    :param pointcloud: Input point cloud.
    :param ax: Matplotlib axis.
    :param cax: Colorbar Axis.
    """
    # colored by height
    im = ax.scatter(-pointcloud[:, 1], pointcloud[:, 0], c=pointcloud[:, 2], s=0.1)
    ax.set_xlim(-16, 16)
    ax.set_ylim(5, 30)
    ax.set_xlabel('y [m]')
    ax.set_xlabel('y [m]')
    ax.set_ylabel('x [m]')
    ax.set_aspect('equal')

    cb = ax.figure.colorbar(im, cax=cax)

    if cblabel == True:
        cb.set_label('Height (m)')

    return cb


def main():
    params = data_preparation.get_default_params()

    params["dataset_path"] = "/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/"
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2,6]
    params["use_npy_cubes"] = False
    params["bev"] = False
    params['label_smoothing'] = False
    params["cfar_folder"] = 'radar_ososos'

    transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = RADCUBE_DATASET(mode='test', transform=transform, params=params)

    camera_projection_matrix, T_camera_lidar = get_sensor_transforms()
    # Example usage
    # roll, pitch, yaw = -2.478, -83.131, 90.419  # Roll, pitch, yaw angles in degrees
    roll, pitch, yaw = 0, -85, 90  # Roll, pitch, yaw angles in degrees
    translation_vector = [0.195, 0.207, -0.482]  # T_camera_lidar[:3, 3]  # Translation vector

    # Create the transformation matrix
    transformation_matrix = create_transformation_matrix(roll, pitch, yaw, translation_vector)
    # transformation_matrix[:, 0] = second_column  # I have no idea why this is needed
    # transformation_matrix[:, 1] = first_column * -1
    # print(transformation_matrix)
    # print(T_camera_lidar)
    distance_from_camera = 30

    # figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 3)
    # make the figure bigger
    fig.set_size_inches(18.5, 10.5)

    counter = 0
    for paths_dict in val_dataset.data_dict.values():

        cam = paths_dict['cam_path']
        lidar = paths_dict['gt_path']
        lidar = lidar.replace('rslidar_points_clean', 'rslidar_points')
        cfar = paths_dict['cfar_path']
        radar = re.sub(r"radar_.+/", r"network/", cfar)

        if not os.path.isfile(radar):
            continue

        counter = counter + 1

        # for cam, lidar, radar, cfar in zip(jpg_files, lidar_files, radar_files,cfar_files):
        # clear the axis
        axs[0, 0].cla()
        axs[0, 1].cla()
        axs[1, 0].cla()
        axs[1, 1].cla()
        axs[0, 2].cla()
        axs[1, 2].cla()
        # cax00.cla()
        # cax01.cla()
        # cax10.cla()
        ##cax11.cla()
        # cax02.cla()
        # cax12.cla()

        divider = make_axes_locatable(axs[0, 0])
        cax00 = divider.append_axes("right", size="5%", pad=0.05)
        divider = make_axes_locatable(axs[0, 1])
        cax01 = divider.append_axes("right", size="5%", pad=0.05)
        divider = make_axes_locatable(axs[1, 0])
        cax10 = divider.append_axes("right", size="5%", pad=0.05)
        divider = make_axes_locatable(axs[1, 1])
        cax11 = divider.append_axes("right", size="5%", pad=0.05)
        divider = make_axes_locatable(axs[0, 2])
        cax02 = divider.append_axes("right", size="5%", pad=0.05)
        divider = make_axes_locatable(axs[1, 2])
        cax12 = divider.append_axes("right", size="5%", pad=0.05)

        axs[0, 0].title.set_text('Lidar')
        axs[0, 1].title.set_text('NN Detector')
        axs[0, 2].title.set_text('CFAR')

        # lidar
        pointcloud = np.load(lidar)
        # pointcloud[:, 1] = -pointcloud[:, 1]
        pointcloud = rf.structured_to_unstructured(pointcloud)
        pointcloud = pointcloud.reshape(-1, 4)

        params = data_preparation.get_default_params()
        # pointcloud = data_preparation.transform_point_cloud(pointcloud, [0, 0, -params['azimuth_offset']],
        #                                          [-params['x_offset'] / 100, -params['y_offset']  / 100, 0])

        uvs, point_depths, filtered_idx = project_pcl_to_image(pointcloud, transformation_matrix,
                                                               camera_projection_matrix, (1216, 1936))
        filtered_by_distance_idx = filter_by_distance(point_depths, distance_from_camera)
        filtered_idx = np.logical_and(filtered_idx, filtered_by_distance_idx)
        uvs = uvs[filtered_idx]

        # plot points on image
        img = plt.imread(cam)
        cb00 = plot_points_on_image(axs[0, 0], cax00, uvs, img, point_depths[filtered_idx], size=7, color_by='depth')

        # plot BEV
        cb10 = plot_bev(pointcloud[filtered_idx], axs[1, 0], cax10, False)

        # RADAR
        radar_points = np.load(radar)
        #radar_points = radar_points[:, :-1]
        pointcloud = radar_points.reshape(-1, 3)
        pointcloud[:, 1] = -pointcloud[:, 1]

        pointcloud = data_preparation.transform_point_cloud(pointcloud, [0, 0, -params['azimuth_offset']],
                                                            [-params['x_offset'] / 100, -params['y_offset'] / 100,
                                                             0])
        uvs, point_depths, filtered_idx = project_pcl_to_image(pointcloud, transformation_matrix,
                                                               camera_projection_matrix, (1216, 1936))
        filtered_by_distance_idx = filter_by_distance(point_depths, distance_from_camera)
        filtered_idx = np.logical_and(filtered_idx, filtered_by_distance_idx)
        uvs = uvs[filtered_idx]

        # plot points on image
        cb01 = plot_points_on_image(axs[0, 1], cax01, uvs, img, point_depths=point_depths[filtered_idx], size=20,
                                    color_by='depth')

        # plot BEV
        cb11 = plot_bev(pointcloud[filtered_idx], axs[1, 1], cax11, False)

        # CFAR
        cfar_points = data_preparation.read_pointcloud(cfar, mode="radar")
        pointcloud = cfar_points
        pointcloud = data_preparation.transform_point_cloud(pointcloud, [0, 0, -params['azimuth_offset']],
                                                            [-params['x_offset'] / 100, -params['y_offset'] / 100,
                                                             0])
        uvs, point_depths, filtered_idx = project_pcl_to_image(pointcloud, transformation_matrix,
                                                               camera_projection_matrix, (1216, 1936))
        filtered_by_distance_idx = filter_by_distance(point_depths, distance_from_camera)
        filtered_idx = np.logical_and(filtered_idx, filtered_by_distance_idx)
        uvs = uvs[filtered_idx]

        # plot points on image
        cb02 = plot_points_on_image(axs[0, 2], cax02, uvs, img, point_depths=point_depths[filtered_idx], size=20,
                                    color_by='depth', alpha=0.5)

        # plot BEV
        cb12 = plot_bev(pointcloud[filtered_idx], axs[1, 2], cax12)

        plt.pause(0.01)
        # fig.show()
        fig.savefig('../frames/' + str(counter) + '.png')

        cb00.remove()
        cb01.remove()
        cb10.remove()
        cb11.remove()
        cb02.remove()
        cb12.remove()

        print("Frame " + str(counter))
        # break


if __name__ == "__main__":
    main()
