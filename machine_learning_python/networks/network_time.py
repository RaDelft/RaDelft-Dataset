import time
import matplotlib.pyplot as plt
#from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

# add parent directory to path
import sys

# apend the absolute path of the parent directory
sys.path.append(sys.path[0] + "/..")
import scipy.io
import re
import os
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_preparation import data_preparation
import torch.nn as nn
from pytorch_lightning.callbacks import RichProgressBar
import plotly.graph_objects as go
from loaders.rad_cube_loader import RADCUBE_DATASET_TIME
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.models as models

OUT_CLASSES = 44  # 44 elevation angles
IN_CHANNELS = 64  # output of the ReduceDNet

# ToDO: Check if goes faster with this:
torch.set_float32_matmul_precision('medium')


# this gets rid of the Doppler dimension to get a "2D image". We go from B*C*D*H*W to B*C*H*W, H and W are ranges and azimuths
class ReduceDNet(nn.Module):
    def __init__(self):
        super(ReduceDNet, self).__init__()

        # Parameters
        in_channels = 2  # Elevation and power
        out_channel_1 = 32  # this can be changed to any number
        out_channel_2 = IN_CHANNELS  # this can be changed to any number, will be the input of next model
        kernel_size1 = (5, 3, 3)
        stride1 = (4, 1, 1)  # (D, H, W), 1/4 of the original size
        padding1 = (2, 1, 1)
        kernel_size2 = (4, 3, 3)
        stride2 = (4, 1, 1)  # (D, H, W), 1/4 of the original size
        padding2 = (1, 1, 1)

        # ToDo: check with Andras, 15 -> 8
        pool_kernel = (8, 1, 1)
        pool_stride = (8, 1, 1)

        # Step 1: Convolution parameters to reduce from 240 to 60
        self.conv1 = nn.Conv3d(in_channels, out_channel_1, kernel_size=kernel_size1, stride=stride1, padding=padding1)
        self.norm1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()

        # Step 2: Convolution parameters to reduce from 60 to 15
        self.conv2 = nn.Conv3d(out_channel_1, out_channel_2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.norm2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()

        # Step 3: Pooling parameters to reduce from 15 to 1
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        # Apply first convolution
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # Apply second convolution
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        # Apply max pooling
        x = self.pool(x)

        return x.squeeze(2)  # Remove the D dimension


class RADPCNET(pl.LightningModule):

    def __init__(self, arch, encoder_name, params, in_channels, out_classes, **kwargs):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.save_hyperparameters()
        self.DopplerReducer = ReduceDNet()

        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        kernel_size = (3, 5, 7)

        self.conv1 = nn.Conv3d(3, 6, kernel_size=kernel_size, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(6, 12, kernel_size=kernel_size, padding='same')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(12, 6, kernel_size=kernel_size, padding='same')
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv3d(6, 3, kernel_size=kernel_size, padding='same')

        #self.model = models.video.r3d_18(pretrained=False, num_classes=out_classes)
        self.counter = 0
        self.params = params
        # preprocessing parameters for image
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # self.params = data_preparation.get_default_params()

    def forward(self, image):
        # Get single frames
        [image1, image2, image3] = torch.chunk(image, 3, axis=1)

        image1 = image1.squeeze(1)
        image2 = image2.squeeze(1)
        image3 = image3.squeeze(1)

        # DopplerReduce Nets
        image1 = self.DopplerReducer(image1)
        image2 = self.DopplerReducer(image2)
        image3 = self.DopplerReducer(image3)

        image1 = image1.float()
        image2 = image2.float()
        image3 = image3.float()

        # Segmentation Model
        mask1 = self.model(image1)
        mask2 = self.model(image2)
        mask3 = self.model(image3)

        mask = torch.stack([mask1, mask2, mask3], 4)

        mask = torch.permute(mask, [0, 4, 1, 2, 3])

        # Temporal smoothing
        mask = self.conv1(mask)
        mask = self.relu1(mask)
        mask = self.conv2(mask)
        mask = self.relu2(mask)
        mask = self.conv3(mask)
        mask = self.relu3(mask)
        mask = self.conv4(mask)

        return mask

    def shared_step(self, batch, stage):
        # Load input and GT
        RAD_cube = batch[0]  # range azimuth doppler cube, the input to the network
        gt_lidar_cube = batch[1]  # TODO here we have to get the gt_cloud and convert it to a mask that fits our loss
        # item_params = batch[2]

        # prepare GT
        # TODO: Check this, I think it is not needed, since we prepare it in the get_item
        # gt_lidar_cube = data_preparation.prepare_lidar_pointcloud(gt_lidar_cube, self.params)

        # Run the network
        RAE_Cube = self.forward(
            RAD_cube)  # output is a binary dense mask of the cube in RAE format: range, azimuth, elevation

        visualize = False

        loss = data_preparation.radarcube_lidarcube_loss_time(RAE_Cube, gt_lidar_cube, self.params)

        if stage == 'valid':
            radar_cube_out = RAE_Cube.sigmoid().squeeze().cpu().detach().numpy()
            radar_cube_out = radar_cube_out > 0.5
            radar_cube_out = radar_cube_out[:, :, :, :-12, 8:-8]
            pd, pfa = data_preparation.compute_pd_pfa(gt_lidar_cube.cpu().detach().numpy(), radar_cube_out)

            return loss, pd, pfa

        return loss

    '''
    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        self.log_dict(metrics, prog_bar=True)
    '''

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=4)
        return loss

    '''
    def on_training_epoch_end(self):
        return self.shared_epoch_end(self.training_step_outputs, "train")
'''

    def validation_step(self, batch, batch_idx):
        loss, pd, pfa = self.shared_step(batch, "valid")

        self.log_dict({'val_loss': loss, 'val_pd': pd, 'val_pfa': pfa, }, on_step=False, on_epoch=True, prog_bar=True,
                      logger=True, batch_size=4)

        # self.log('val_loss', loss, 'val_pd', pd, 'val_pfa', pfa, prog_bar=True, batch_size=4)
        return loss

    '''
    def on_validation_epoch_end(self):
        print('\n')
        return self.shared_epoch_end(self.validation_step_outputs, "valid")
'''

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    '''
    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")
'''

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler,
                'monitor': 'train_loss_epoch'}  # I changed the monitored topic  IGNA because it died


# main function
def main(params):
    transform = transforms.Compose([transforms.ToTensor()])

    # Create training and validation datasets
    train_dataset = RADCUBE_DATASET_TIME(mode='train', transform=transform, params=params)
    val_dataset = RADCUBE_DATASET_TIME(mode='val', transform=transform, params=params)

    # Create training and validation data loaders
    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=False)
    model = RADPCNET("FPN", "resnet18", params, in_channels=IN_CHANNELS, out_classes=OUT_CLASSES)

    #path = 'lightning_logs/version_44/checkpoints/epoch=4-step=8190.ckpt'
    # path = 'lightning_logs/version_6/checkpoints/epoch=27-step=69328.ckpt'
    # NOTE file is not always readable, permissions can be fucked
    #checkpoint = torch.load(path)
    #model.load_state_dict(checkpoint['state_dict'])

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=60,
        callbacks=[checkpoint_callback, RichProgressBar(leave=True, theme=RichProgressBarTheme(metrics_format='.4e'))],
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


def test(params):
    # Load model
    path = '../logs/lightning_logs/version_18/checkpoints/epoch=39-step=65520.ckpt'
    #path = 'lightning_logs/version_6/checkpoints/epoch=27-step=69328.ckpt'
    # NOTE file is not always readable, permissions can be fucked
    checkpoint = torch.load(path)
    model = RADPCNET("deeplabv3plus", "resnet18", params, in_channels=IN_CHANNELS, out_classes=OUT_CLASSES)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Create Loader
    transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = RADCUBE_DATASET_TIME(mode='test', transform=transform, params=params)

    # Create training and validation data loaders
    num_workers = 16
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=False)

    for batch in val_loader:

        radar_cube, lidar_cube, data_dict = batch

        with torch.no_grad():
            output = model(radar_cube)
            for i in range(lidar_cube.shape[0]):
                for t in range(lidar_cube.shape[1]):
                    #a, b = data_preparation.compute_pd_pfa(lidar_cube[0, :, :, :].cpu().numpy(), output[i, :, :, :])

                    output_t = output[i, t, :, :, :]
                    data_dict_t = data_dict[t]

                    radar_pc = data_preparation.cube_to_pointcloud(output_t, params, radar_cube[i, t, :, :, :],
                                                                   data_dict_t["elevation_path"][i], 'radar', False,
                                                                   data_dict_t["power_path"][i], )

                    radar_pc[:, 2] = -radar_pc[:, 2]

                    cfar_path = data_dict_t["cfar_path"][i]
                    save_path = re.sub(r"radar_.+/", r"network/", cfar_path)
                    print(save_path)

                    np.save(save_path, radar_pc)


def compute_metrics(params):
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
            radar_distance = radar_distance + data_preparation.compute_chamfer_distance(lidarpc, radarpc)

            lidar_cube = data_preparation.lidarpc_to_lidarcube(lidarpc, params)
            #cfar_cube = data_preparation.lidarpc_to_lidarcube(cfarpc,params)

            radar_cube = data_preparation.lidarpc_to_lidarcube(radarpc, params)

            #pd_cfar_aux, pfa_cfar_aux = data_preparation.compute_pd_pfa(lidar_cube, cfar_cube)
            pd_radar_aux, pfa_radar_aux = data_preparation.compute_pd_pfa(lidar_cube, radar_cube)

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


if __name__ == "__main__":
    params = data_preparation.get_default_params()

    # if Andras is running the code, change ROS_DS_path to his path
    if os.path.exists("/home/apalffy"):
        params["ROS_DS_Path"] = "/DATA/Datasets/EWI_IV_recordings/ROS_DS/_2023-10-09-15-04-36"
        # assume that radar cubes are one level above ROS_DS_Path. strip the last folder, add radar_cubes
        parent_of_ros_ds = os.path.dirname(params["ROS_DS_Path"])
        parent_of_that = os.path.dirname(parent_of_ros_ds)
        params["radar_cubes"] = os.path.join(parent_of_that, "radar_cubes")
        params["use_npy_cubes"] = True

    if os.path.exists("/home/iroldan"):
        params["ROS_DS_Path"] = "/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment1/rosDS"
        params["radar_cubes"] = "/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/Day2Experiment1/RadarCubes"
        params["dataset_path"] = "/media/iroldan/179bc4e0-0daa-4d2d-9271-25c19bcfd403/"
        params["train_val_scenes"] = [1, 3, 4, 5, 7]
        params["test_scenes"] = [2, 6]
        #params["test_scenes"] = [6]
        params["use_npy_cubes"] = False

    params["train_test_split_percent"] = 0.8
    params["bev"] = False
    params["cfar_folder"] = 'radar_ososos'

    #main(params)
    test(params)

    #params['label_smoothing'] = False
    compute_metrics(params)

    # Dataset statistics
    '''
    start_time = time.time()
    dataset = RADCUBE_DATASET_MULTI(mode='train', transform=None, params=params)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=16)

    sparseness = torch.zeros(len(data_loader))
    count = 0
    for batch in data_loader:
        _, lidar_cube, _ = batch

        n_zeros = torch.sum(lidar_cube == 0)

        sparseness[count] = (n_zeros / (lidar_cube.size()[0] *lidar_cube.size()[1] * lidar_cube.size()[2] * lidar_cube.size()[3]))
        count = count + 1

    mean_sparness = torch.mean(sparseness)
    print('MEAN SPARNESS: ' + str(mean_sparness))
    print('TIME: ' + str(time.time() - start_time))
    '''
