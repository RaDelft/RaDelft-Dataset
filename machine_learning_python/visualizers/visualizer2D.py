from loader.rad_cube_loader import RADCUBE_DATASET_MULTI
from data_preprocessing import data_preparation
import torchvision.transforms as transforms
import os
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import dash
from dash import Dash, dcc, html, Input, Output, callback


@callback(Output('graph_point_clouds', 'figure'), Input('frame', 'value'))
def update_graph_live(n):

    paths_dict = val_dataset.data_dict[n]

    lidar = paths_dict['gt_path']
    cfar = paths_dict['cfar_path']
    radar = re.sub(r"radar_.+/", r"network2D/", cfar)

    # Load Lidar
    lidar_pc = np.load(lidar)

    # Load CFAR
    cfar_pc = data_preparation.read_pointcloud(cfar, mode="radar")

    # Load NN detector
    radar_pc = np.load(radar)
    radar_pc[:, 1] = -radar_pc[:, 1]
    # pointcloud[:, 1] = -pointcloud[:, 1]

    traceLidar = go.Scatter(x=lidar_pc[:, 1], y=lidar_pc[:, 0],
                              mode='markers', marker=dict(size=2, color='blue', opacity=0.8))

    traceCFAR = go.Scatter(x=cfar_pc[:, 1], y=cfar_pc[:, 0],
                             mode='markers', marker_symbol='cross', marker=dict(size=2, color='red', opacity=0.2))

    traceRadar = go.Scatter(x=radar_pc[:, 1], y=radar_pc[:, 0],
                             mode='markers', marker_symbol='cross', marker=dict(size=2, color='red', opacity=0.2))



    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.01,specs=[[{'type': 'scatter'}], [{'type': 'scatter'}]])

    fig.add_trace(traceLidar, 1, 1)
    fig.add_trace(traceRadar, 1, 1)

    fig.add_trace(traceLidar, 2, 1)
    fig.add_trace(traceCFAR, 2, 1)

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-2, y=0, z=1.5)
    )
    '''
    fig.layout.scene.camera.up = dict(x=0, y=0, z=1)
    fig.layout.scene.camera.center = dict(x=0, y=0, z=0)
    fig.layout.scene.camera.eye = dict(x=-2, y=0, z=1.5)

    fig.layout.scene2.camera.up = dict(x=0, y=0, z=1)
    fig.layout.scene2.camera.center = dict(x=0, y=0, z=0)
    fig.layout.scene2.camera.eye = dict(x=-2, y=0, z=1.5)

    fig.layout.scene.aspectmode = "data"
    fig.layout.scene2.aspectmode = "data"
'''


    return fig

if __name__ == "__main__":
    params = data_preparation.get_default_params()

    params["dataset_path"] = "/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/"
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2]
    params["use_npy_cubes"] = False
    params["bev"] = False
    params['label_smoothing'] = False
    params["cfar_folder"] = 'radar_ososos2D'

    transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = RADCUBE_DATASET_MULTI(mode='test', transform=transform, params=params)
    count = 1
    #fig = go.FigureWidget()
    app = dash.Dash()
    app.layout = html.Div([
        "Frame: ", dcc.Input(id="frame", value=0, type="number"),
        html.Div([dcc.Graph(id='graph_point_clouds', style={'height': '95vh'})])
    ])

    app.run_server()
