
import torchvision.transforms as transforms
import numpy as np
from data_preparation import data_preparation
from loaders.rad_cube_loader import RADCUBE_DATASET_TIME,RADCUBE_DATASET_MULTI
from sklearn.neighbors import KDTree

def compute_metrics(params):

    # Create Loader
    transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = RADCUBE_DATASET_MULTI(mode='test', transform=transform, params=params)

    cfar_distance = 0
    radar_distance = 0
    count = 0
    pd_cfar = 0
    pd_radar = 0
    pfa_cfar = 0
    pfa_radar = 0
    for dict in val_dataset.data_dict.values():
        lidar = dict['gt_path']
        cfar = dict['cfar_path']
        network_output = cfar.replace('radar_ososos', 'network')

        lidarpc = np.load(lidar)
        #cfarpc = data_preparation.read_pointcloud(cfar, mode="radar")
        #cfarpc = cfarpc[:,0:3]

        radarpc = np.load(network_output)
        radarpc[:, 1] = -radarpc[:, 1]

        #cfar_distance = cfar_distance + data_preparation.compute_chamfer_distance(lidarpc,cfarpc)
        radar_distance = radar_distance + compute_chamfer_distance(lidarpc, radarpc)

        lidar_cube = data_preparation.lidarpc_to_lidarcube(lidarpc, params)
        #cfar_cube = data_preparation.lidarpc_to_lidarcube(cfarpc,params)

        radar_cube = data_preparation.lidarpc_to_lidarcube(radarpc,params)

        #pd_cfar_aux, pfa_cfar_aux = data_preparation.compute_pd_pfa(lidar_cube, cfar_cube)
        pd_radar_aux , pfa_radar_aux = compute_pd_pfa(lidar_cube, radar_cube)

        #pd_cfar = pd_cfar + pd_cfar_aux
        #pfa_cfar = pfa_cfar + pfa_cfar_aux
        pd_radar = pd_radar + pd_radar_aux
        pfa_radar = pfa_radar + pfa_radar_aux


        count = count + 1

        if count % 10 == 0:
            print(str(count))

    print('Pd CFAR: ' + str(pd_cfar/count))
    print('Pd NET: ' + str(pd_radar / count))
    print('----------')
    print('Pfa CFAR: ' + str(pfa_cfar / count))
    print('Pfa Net: ' + str(pfa_radar / count))
    print('----------')
    print('Distance CFAR: ' + str(cfar_distance / count))
    print('Distance Net: ' + str(radar_distance / count))
def compute_metrics_time(params):
    # Create Loader
    transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = RADCUBE_DATASET_TIME(mode='test', transform=transform, params=params)

    cfar_distance = 0
    radar_distance = 0
    count = 0
    pd_cfar = 0
    pd_radar = 0
    pfa_cfar = 0
    pfa_radar = 0
    for dict in val_dataset.data_dict.values():
        for t in dict.keys():
            lidar = dict[t]['gt_path']
            cfar = dict[t]['cfar_path']
            network_output = cfar.replace('radar_ososos', 'network')

            lidarpc = np.load(lidar)
            #cfarpc = data_preparation.read_pointcloud(cfar, mode="radar")
            #cfarpc = cfarpc[:,0:3]

            radarpc = np.load(network_output)
            radarpc[:, 1] = -radarpc[:, 1]

            #cfar_distance = cfar_distance + data_preparation.compute_chamfer_distance(lidarpc,cfarpc)
            radar_distance = radar_distance + compute_chamfer_distance(lidarpc, radarpc)

            lidar_cube = data_preparation.lidarpc_to_lidarcube(lidarpc, params)
            #cfar_cube = data_preparation.lidarpc_to_lidarcube(cfarpc,params)

            radar_cube = data_preparation.lidarpc_to_lidarcube(radarpc, params)

            #pd_cfar_aux, pfa_cfar_aux = data_preparation.compute_pd_pfa(lidar_cube, cfar_cube)
            pd_radar_aux, pfa_radar_aux = compute_pd_pfa(lidar_cube, radar_cube)

            #pd_cfar = pd_cfar + pd_cfar_aux
            #pfa_cfar = pfa_cfar + pfa_cfar_aux
            pd_radar = pd_radar + pd_radar_aux
            pfa_radar = pfa_radar + pfa_radar_aux

            count = count + 1

            if count % 10 == 0:
                print(str(count))

    print('Pd CFAR: ' + str(pd_cfar / count))
    print('Pd NET: ' + str(pd_radar / count))
    print('----------')
    print('Pfa CFAR: ' + str(pfa_cfar / count))
    print('Pfa Net: ' + str(pfa_radar / count))
    print('----------')
    print('Distance CFAR: ' + str(cfar_distance / count))
    print('Distance Net: ' + str(radar_distance / count))

def compute_chamfer_distance(point_cloud1, point_cloud2):

    num_points1 = point_cloud1.shape[1]
    num_points2 = point_cloud2.shape[1]

    tree1 = KDTree(point_cloud1, leaf_size=num_points1+1, metric='euclidean')
    tree2 = KDTree(point_cloud2, leaf_size=num_points2+1,metric='euclidean')
    distances1, _ = tree1.query(point_cloud2)
    distances2, _ = tree2.query(point_cloud1)
    av_dist1 = np.sum(distances1) / np.size(distances1)
    av_dist2 = np.sum(distances2) / np.size(distances2)
    dist = av_dist1 + av_dist2

    return dist

def compute_pd_pfa(ground_truth, prediction):

    # Flatten the matrices to 1D arrays
    ground_truth_flat = ground_truth.flatten()
    prediction_flat = prediction.flatten()

    # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.sum((ground_truth_flat == 1) & (prediction_flat == 1))
    FP = np.sum((ground_truth_flat == 0) & (prediction_flat == 1))
    FN = np.sum((ground_truth_flat == 1) & (prediction_flat == 0))

    # Compute True Positive Rate (TPR) and False Positive Rate (FPR)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + (ground_truth_flat.size - TP - FN)) if (FP + (ground_truth_flat.size - TP - FN)) > 0 else 0

    return TPR, FPR