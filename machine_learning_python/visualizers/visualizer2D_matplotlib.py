from loaders.rad_cube_loader import RADCUBE_DATASET_MULTI
from data_preparation import data_preparation
import torchvision.transforms as transforms
import re
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    params = data_preparation.get_default_params()

    params["dataset_path"] = "/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/"
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2]
    params["use_npy_cubes"] = False
    params["bev"] = True
    params['label_smoothing'] = False
    params["cfar_folder"] = 'radar_ososos2D'

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
        cfar_pc = data_preparation.read_pointcloud(cfar, mode="radar")
        cfar_pc[:, 1] = -cfar_pc[:, 1]
        cfar_pc = data_preparation.transform_point_cloud(cfar_pc, [0, 0, params['azimuth_offset']],
                                                          [-params['x_offset'] / 100, -params['y_offset'] / 100,
                                                           0])

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
        lidar_pc = lidar_pc[lidar_pc[:, 1] > -30, :]
        lidar_pc = lidar_pc[lidar_pc[:, 1] < 30, :]
        cfar_pc = cfar_pc[cfar_pc[:, 1] > -30, :]
        cfar_pc = cfar_pc[cfar_pc[:, 1] < 30, :]

        cam = paths_dict['cam_path']
        img = plt.imread(cam)
        img = img[500:-150,:,:]
        layout = [
            ["camera", "camera","camera"],
            ["radar", "lidar", "cfar"]
        ]
        fig, axd = plt.subplot_mosaic(layout, figsize=(12, 8), layout='constrained')
        axd["camera"].imshow(img, aspect='0.9')
        axd["camera"].axis('off')
        axd["radar"].scatter(radar_pc[:, 1], radar_pc[:, 0], s=1)

        axd["radar"].set_xlim(-30, 30)
        axd["radar"].set_title('Radar + NN')
        #plt.gca().invert_xaxis()
        axd["radar"].set_xlabel('x (m)')
        axd["radar"].set_ylabel('y (m)')
        #cbar=plt.colorbar(radar_plot, fraction=0.025, pad=0)
        #cbar.set_label('Height (m)')

        axd["lidar"].scatter(lidar_pc[:, 1], lidar_pc[:, 0], s=1)
        axd["lidar"].set_xlim(-30, 30)
        axd["lidar"].set_xlabel('x (m)')
        axd["lidar"].set_ylabel('y (m)')
        axd["lidar"].set_title('Lidar')
        #plt.gca().invert_xaxis()
        #cbar = plt.colorbar(lidar_plot, fraction=0.025, pad=0)
        #cbar.set_label('Height (m)')

        axd["cfar"].scatter(cfar_pc[:, 1], cfar_pc[:, 0], s=1)
        axd["cfar"].set_xlim(-30, 30)
        axd["cfar"].set_xlabel('x (m)')
        axd["cfar"].set_ylabel('y (m)')
        axd["cfar"].set_title('CFAR')

        #plt.tight_layout()
        #plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.savefig('../frames/' + str(count) + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        print("Frame " + str(count))
        count = count + 1
