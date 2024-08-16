import numpy as np
from data_preparation import data_preparation
from loaders.rad_cube_loader import RADCUBE_DATASET_TIME, RADCUBE_DATASET
from sklearn.neighbors import KDTree


def compute_metrics(params):
    """
    Compute the Chamfer distance and the Pfa and Pd on all the frames.
    This is used for the single frame version.

    :param params: default parameters from data_preparation.py
    :return: Nothing, only prints the result
    """
    # Create Loader
    val_dataset = RADCUBE_DATASET(mode='test', params=params)

    cfar_distance = 0
    radar_distance = 0
    count = 0
    pd_cfar = 0
    pd_radar = 0
    pfa_cfar = 0
    pfa_radar = 0

    for dataset_dict in val_dataset.data_dict.values():

        # Generate lidar_cube
        lidar = dataset_dict['gt_path']
        lidarpc = np.load(lidar)
        lidar_cube = data_preparation.lidarpc_to_lidarcube(lidarpc, params)

        # Load radar point clouds
        cfar = dataset_dict['cfar_path']
        network_output = cfar.replace('radar_ososos', 'network')
        radarpc = np.load(network_output)
        if radarpc.shape[1] == 4:
            radarpc = radarpc[:, :-1]  # Remove speed to compute metrics
        radarpc[:, 1] = -radarpc[:, 1]

        # Compute Metrics
        radar_distance = radar_distance + compute_chamfer_distance(lidarpc, radarpc)
        radar_cube = data_preparation.lidarpc_to_lidarcube(radarpc, params)
        pd_radar_aux, pfa_radar_aux = compute_pd_pfa(lidar_cube, radar_cube)
        pd_radar = pd_radar + pd_radar_aux
        pfa_radar = pfa_radar + pfa_radar_aux

        '''
        Uncomment if the CFAR metrics wants to be calculated
        
        cfarpc = data_preparation.read_pointcloud(cfar, mode="radar")
        cfarpc = cfarpc[:,0:3]
        cfar_distance = cfar_distance + data_preparation.compute_chamfer_distance(lidarpc,cfarpc)
        cfar_cube = data_preparation.lidarpc_to_lidarcube(cfarpc,params)
        pd_cfar_aux, pfa_cfar_aux = data_preparation.compute_pd_pfa(lidar_cube, cfar_cube)
        pd_cfar = pd_cfar + pd_cfar_aux
        pfa_cfar = pfa_cfar + pfa_cfar_aux
        '''

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


def compute_metrics_time(params):
    """
    Compute the Chamfer distance and the Pfa and Pd on all the frames.
    This is used for the multi frame version.

    :param params: default parameters from data_preparation.py
    :return: Nothing, only prints the result
    """
    # Create Loader
    val_dataset = RADCUBE_DATASET_TIME(mode='test',  params=params)

    cfar_distance = 0
    radar_distance = 0
    count = 0
    pd_cfar = 0
    pd_radar = 0
    pfa_cfar = 0
    pfa_radar = 0

    for dataset_dict in val_dataset.data_dict.values():
        for t in dataset_dict.keys():

            # Generate lidar_cube
            lidar = dataset_dict[t]['gt_path']
            lidarpc = np.load(lidar)
            lidarpc[:, 1] = -lidarpc[:, 1]

            lidar_cube = data_preparation.lidarpc_to_lidarcube(lidarpc, params)

            # Load radar point clouds
            cfar = dataset_dict[t]['cfar_path']
            network_output = cfar.replace('radar_ososos', 'network')
            radarpc = np.load(network_output)
            if radarpc.shape[1] == 4:
                radarpc = radarpc[:, :-1]           #Remove speed to compute metrics

            # Compute Metrics
            radar_distance = radar_distance + compute_chamfer_distance(lidarpc, radarpc)
            radar_cube = data_preparation.lidarpc_to_lidarcube(radarpc, params)
            pd_radar_aux, pfa_radar_aux = compute_pd_pfa(lidar_cube, radar_cube)
            pd_radar = pd_radar + pd_radar_aux
            pfa_radar = pfa_radar + pfa_radar_aux

            '''
            Uncomment if the CFAR metrics wants to be calculated
            
            cfarpc = data_preparation.read_pointcloud(cfar, mode="radar")
            cfarpc = cfarpc[:,0:3]
            cfar_distance = cfar_distance + compute_chamfer_distance(lidarpc,cfarpc)
            cfar_cube = data_preparation.lidarpc_to_lidarcube(cfarpc,params)
            pd_cfar_aux, pfa_cfar_aux = compute_pd_pfa(lidar_cube, cfar_cube)
            pd_cfar = pd_cfar + pd_cfar_aux
            pfa_cfar = pfa_cfar + pfa_cfar_aux
            '''
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
    """
    Compute the Chamfer distance between two set of points

    :param point_cloud1: the first set of points
    :param point_cloud2: the second set of points
    :return: the Chamfer distance
    """
    tree1 = KDTree(point_cloud1, metric='euclidean')
    tree2 = KDTree(point_cloud2, metric='euclidean')
    distances1, _ = tree1.query(point_cloud2)
    distances2, _ = tree2.query(point_cloud1)
    av_dist1 = np.sum(distances1) / np.size(distances1)
    av_dist2 = np.sum(distances2) / np.size(distances2)
    dist = av_dist1 + av_dist2

    return dist


def compute_pd_pfa(ground_truth, prediction):
    """
    Compute the Pd and Pfa between two 3D cubes

    :param ground_truth: the 3D cube to compare. Usually the lidar cube.
    :param prediction: the estimated 3D cube. Usually the rada cube.
    :return: the Pd and Pfa
    """
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
