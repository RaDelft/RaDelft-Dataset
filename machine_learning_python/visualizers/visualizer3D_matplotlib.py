from loaders.rad_cube_loader import RADCUBE_DATASET
from data_preparation import data_preparation
import re
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    params = data_preparation.get_default_params()

    params["dataset_path"] = "/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/"
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [6]
    params["bev"] = False
    params["cfar_folder"] = 'radar_ososos'

    val_dataset = RADCUBE_DATASET(mode='test',  params=params)
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
        #radar_pc[:, 1] = -radar_pc[:, 1]

        radar_pc = data_preparation.transform_point_cloud(radar_pc, [0, 0, params['azimuth_offset']],
                                                          [-params['x_offset'] / 100, -params['y_offset'] / 100,
                                                           0])
        #radar_pc[:, 1] = -radar_pc[:, 1]
        # Trim for plotting
        radar_pc = radar_pc[radar_pc[:, 1] > -30, :]
        radar_pc = radar_pc[radar_pc[:, 1] < 30, :]
        radar_pc = radar_pc[radar_pc[:, 2] < 10, :]
        lidar_pc = lidar_pc[lidar_pc[:, 1] > -30, :]
        lidar_pc = lidar_pc[lidar_pc[:, 1] < 30, :]
        lidar_pc = lidar_pc[lidar_pc[:, 2] < 10, :]
        radar_pc = radar_pc[radar_pc[:, 2] > -1, :]
        lidar_pc = lidar_pc[lidar_pc[:, 2] > -1, :]



        layout = [
            ["radar", "lidar"]
        ]

        fig, axd = plt.subplot_mosaic(layout, figsize=(16, 10), layout='constrained',
                                      per_subplot_kw={('radar', 'lidar'): {'projection': '3d'}},
                                      gridspec_kw={'wspace': 0.01, 'hspace': 0.01},
                                      )



        radarplot = axd["radar"].scatter(radar_pc[:, 0], radar_pc[:, 1], radar_pc[:, 2], c=radar_pc[:, 2], alpha=0.8, s=1)
        #axd["radar"].invert_xaxis()
        axd["radar"].view_init(elev=30, azim=0)
        axd["radar"].set_ylim(-30, 30)
        axd["radar"].set_zlim(-2, 10)
        axd["radar"].set_xlim(0, 50)
        axd["radar"].set_box_aspect([2, 2, 0.5])

        axd["radar"].set_xlabel('y (m)')
        axd["radar"].set_ylabel('x (m)')
        axd["radar"].set_zlabel('z (m)')
        axd["radar"].invert_xaxis()
        cbar=plt.colorbar(radarplot, fraction=0.025, pad=0)
        cbar.set_label('Height (m)')

        #ax = fig.add_subplot(1,2,2,projection='3d')
        lidarplot=axd["lidar"].scatter(lidar_pc[:, 0], lidar_pc[:, 1], lidar_pc[:, 2], c=lidar_pc[:, 2], alpha=0.8, s=1)
        axd["lidar"].view_init(elev=30, azim=0)
        axd["lidar"].set_ylim(-30, 30)
        axd["lidar"].set_zlim(-2, 10)
        axd["lidar"].set_xlim(0, 50)
        axd["lidar"].set_box_aspect([2, 2, 0.5])
        #axd["lidar"].set_aspect('equal', adjustable='box')
        axd["lidar"].set_xlabel('y (m)')
        axd["lidar"].set_ylabel('x (m)')
        axd["lidar"].set_zlabel('z (m)')
        #axd["lidar"].set_title('Lidar')
        axd["lidar"].invert_xaxis()
        cbar = plt.colorbar(lidarplot, fraction=0.025, pad=0)
        cbar.set_label('Height (m)')


        #plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.savefig('../frames/' + str(count).zfill(5) + '.png')#, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close(fig)

        cam = paths_dict['cam_path']
        img = plt.imread(cam)
        img = img[500:, :, :]
        fig, axCamera = plt.subplots(figsize=(16, 10))
        plt.imshow(img, aspect='0.9')
        axCamera.axis('off')
        fig.savefig('../frames/camera_' + str(count).zfill(5) + '.png')  # , bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close(fig)

        print("Frame " + str(count))
        count = count + 1
