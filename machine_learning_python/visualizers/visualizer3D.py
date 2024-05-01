from loaders.rad_cube_loader import RADCUBE_DATASET
from data_preparation import data_preparation
import torchvision.transforms as transforms
import os
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import dash
from dash import Dash, dcc, html, Input, Output, callback
import torch

@callback(Output('graph_point_clouds', 'figure'), Input('frame', 'value'))
def update_graph_live(n):

    paths_dict = val_dataset.data_dict[n]

    lidar = paths_dict['gt_path']
    cfar = paths_dict['cfar_path']
    radar = re.sub(r"radar_.+/", r"network/", cfar)

    # Load Lidar
    lidar_pc = np.load(lidar)
    lidar_cube = data_preparation.lidarpc_to_lidarcube(lidar_pc, None)
    lidar_cube = torch.from_numpy(lidar_cube)

    lidar_pc_low = data_preparation.cube_to_pointcloud(lidar_cube, None, None, None, mode='lidar')

    # Load CFAR
    #cfar_pc = data_preparation.read_pointcloud(cfar, mode="radar")

    # Load NN detector
    radar_pc = np.load(radar)
    radar_pc[:, 1] = -radar_pc[:, 1]


    traceLidar = go.Scatter3d(x=lidar_pc[:, 0], y=lidar_pc[:, 1], z=lidar_pc[:, 2],
                              mode='markers', marker=dict(size=2, color=lidar_pc[:, 2], colorscale='Viridis', opacity=0.8))

    #traceCFAR = go.Scatter3d(x=cfar_pc[:, 0], y=cfar_pc[:, 1], z=-cfar_pc[:, 2],
    #                         mode='markers', marker_symbol='cross', marker=dict(size=2, color='red', opacity=1))

    traceRadar = go.Scatter3d(x=radar_pc[:, 0], y=radar_pc[:, 1], z=radar_pc[:, 2],
                             mode='markers', marker_symbol='cross', marker=dict(size=2, color=radar_pc[:, 2], colorscale='Viridis', opacity=0.2,colorbar=dict(thickness=20)))



    fig = make_subplots(rows=1, cols=2, vertical_spacing=0.01,specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

    #fig.add_trace(traceLidar, 1, 1)
    fig.add_trace(traceRadar, 1, 1)

    fig.add_trace(traceLidar, 1, 2)
    #fig.add_trace(traceCFAR, 2, 1)

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-2, y=0, z=1.5)
    )

    #fig.layout.scene.camera.up = dict(x=0, y=0, z=1)
    #fig.layout.scene.camera.center = dict(x=0, y=0, z=0)
    fig.layout.scene.camera.eye = dict(x=-2, y=0, z=1.5)



    #fig.layout.scene2.camera.up = dict(x=0, y=0, z=1)
    #fig.layout.scene2.camera.center = dict(x=0, y=0, z=0)
    fig.layout.scene2.camera.eye = dict(x=-2, y=0, z=1.5)

    fig.layout.scene.aspectmode = "data"
    fig.layout.scene2.aspectmode = "data"

    fig.layout.scene.yaxis = dict(range=[-30, 30])
    fig.layout.scene2.yaxis = dict(range=[-30, 30])

    return fig

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
    val_dataset = RADCUBE_DATASET(mode='test', transform=transform, params=params)
    count = 1
    #fig = go.FigureWidget()
    app = dash.Dash()
    app.layout = html.Div([
        "Frame: ", dcc.Input(id="frame", value=0, type="number"),
        html.Div([dcc.Graph(id='graph_point_clouds', style={'height': '95vh'})])
    ])

    #app.run_server(debug='True')
    app.run_server()
