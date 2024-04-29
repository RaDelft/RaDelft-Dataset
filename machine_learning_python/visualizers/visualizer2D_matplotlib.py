from loaders.rad_cube_loader import RADCUBE_DATASET_MULTI
from data_preparation import data_preparation
import torchvision.transforms as transforms
import re
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__":
    params = data_preparation.get_default_params()

    params["dataset_path"] = "/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/"
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [6]
    params["use_npy_cubes"] = False
    params["bev"] = False
    params['label_smoothing'] = False
    params["cfar_folder"] = 'radar_ososos'

    transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = RADCUBE_DATASET_MULTI(mode='test', transform=transform, params=params)
    count = 1

    for paths_dict in val_dataset.data_dict.values():
        # Load Data
        lidar = paths_dict['gt_path']
        cfar = paths_dict['cfar_path']
        radar = re.sub(r"radar_.+/", r"network/", cfar)

        # Load Lidar
        lidar_pc = np.load(lidar)
        lidar_pc[:, 1] = -lidar_pc[:, 1]
        #lidar_cube = data_preparation.lidarpc_to_lidarcube(lidar_pc, None)
        #lidar_cube = torch.from_numpy(lidar_cube)
        lidar_pc = data_preparation.transform_point_cloud(lidar_pc, [0, 0, params['azimuth_offset']],
                                                          [-params['x_offset'] / 100, -params['y_offset'] / 100,
                                                           0])

        #lidar_pc_low = data_preparation.cube_to_pointcloud(lidar_cube, None, None, None, mode='lidar')

        # Load CFAR
        # cfar_pc = data_preparation.read_pointcloud(cfar, mode="radar")

        # Load NN detector
        radar_pc = np.load(radar)
        radar_pc[:, 1] = -radar_pc[:, 1]


        radar_pc = data_preparation.transform_point_cloud(radar_pc, [0, 0, -params['azimuth_offset']],
                                                          [-params['x_offset'] / 100, -params['y_offset'] / 100,
                                                           0])
        radar_pc[:, 1] = -radar_pc[:, 1]
        # Trim for plotting
        radar_pc = radar_pc[radar_pc[:, 1] > -30, :]
        radar_pc = radar_pc[radar_pc[:, 1] < 30, :]
        radar_pc = radar_pc[radar_pc[:, 2] < 10, :]
        lidar_pc = lidar_pc[lidar_pc[:, 1] > -30, :]
        lidar_pc = lidar_pc[lidar_pc[:, 1] < 30, :]
        lidar_pc = lidar_pc[lidar_pc[:, 2] < 10, :]
        radar_pc = radar_pc[radar_pc[:, 2] > -1, :]
        lidar_pc = lidar_pc[lidar_pc[:, 2] > -1, :]


        fig = plt.figure(figsize=(16, 8))

        ax = fig.add_subplot(1,2,1,projection='3d')
        radar_plot = ax.scatter(radar_pc[:, 0], radar_pc[:, 1], radar_pc[:, 2], c=radar_pc[:, 2], alpha=0.8, s=1)
        ax.view_init(elev=30, azim=0)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-2, 10)
        ax.set_xlim(0, 50)
        ax.set_box_aspect([2, 2, 0.5])
        #ax.set_title('Radar + NN')
        plt.gca().invert_xaxis()
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        cbar=plt.colorbar(radar_plot, fraction=0.025, pad=0)
        cbar.set_label('Height (m)')

        ax = fig.add_subplot(1,2,2,projection='3d')
        lidar_plot = ax.scatter(lidar_pc[:, 0], lidar_pc[:, 1], lidar_pc[:, 2], c=lidar_pc[:, 2], alpha=0.8, s=1)
        ax.view_init(elev=30, azim=0)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-2, 10)
        ax.set_xlim(0, 50)
        ax.set_box_aspect([2, 2, 0.5])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        #ax.set_title('Lidar')
        plt.gca().invert_xaxis()
        cbar = plt.colorbar(lidar_plot, fraction=0.025, pad=0)
        cbar.set_label('Height (m)')

        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.savefig('../frames/' + str(count) + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        print("Frame " + str(count))
        count = count + 1
