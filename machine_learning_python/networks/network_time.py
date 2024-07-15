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
from loaders.rad_cube_loader import RADCUBE_DATASET_TIME
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.models as models
from utils.compute_metrics import compute_metrics_time, compute_pd_pfa

OUT_CLASSES = 44  # 44 elevation angles
IN_CHANNELS = 64  # output of the ReduceDNet

# ToDO: Check if goes faster with this:
torch.set_float32_matmul_precision('medium')


# this gets rid of the Doppler dimension to get a "2D image".
# We go from B*C*D*H*W to B*C*H*W, H and W are ranges and azimuths
class DopplerEncoder(nn.Module):
    def __init__(self):
        super(DopplerEncoder, self).__init__()

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


class NeuralNetworkRadarDetector(pl.LightningModule):

    def __init__(self, arch, encoder_name, params, in_channels, out_classes, **kwargs):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.save_hyperparameters()
        self.DopplerReducer = DopplerEncoder()

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

        self.counter = 0
        self.params = params

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

        # Run the network
        RAE_Cube = self.forward(
            RAD_cube)  # output is a binary dense mask of the cube in RAE format: range, azimuth, elevation

        loss = data_preparation.radarcube_lidarcube_loss_time(RAE_Cube, gt_lidar_cube, self.params)

        if stage == 'valid':
            radar_cube_out = RAE_Cube.sigmoid().squeeze().cpu().detach().numpy()
            radar_cube_out = radar_cube_out > 0.5
            radar_cube_out = radar_cube_out[:, :, :, :-12, 8:-8]
            pd, pfa = compute_pd_pfa(gt_lidar_cube.cpu().detach().numpy(), radar_cube_out)

            return loss, pd, pfa

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=4)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pd, pfa = self.shared_step(batch, "valid")

        self.log_dict({'val_loss': loss, 'val_pd': pd, 'val_pfa': pfa, }, on_step=False, on_epoch=True, prog_bar=True,
                      logger=True, batch_size=4)

        return loss

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler,
                'monitor': 'train_loss_epoch'}


# main function
def main(params):
    # Create training and validation datasets
    train_dataset = RADCUBE_DATASET_TIME(mode='train', params=params)
    val_dataset = RADCUBE_DATASET_TIME(mode='val', params=params)

    # Create training and validation data loaders
    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=False)
    model = NeuralNetworkRadarDetector("FPN", "resnet18", params, in_channels=IN_CHANNELS, out_classes=OUT_CLASSES)

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


def generate_point_clouds(params):
    # Load model
    path = 'lightning_logs/version_10/checkpoints/epoch=15-step=26208.ckpt'
    checkpoint = torch.load(path)
    model = NeuralNetworkRadarDetector("FPN", "resnet18", params, in_channels=IN_CHANNELS, out_classes=OUT_CLASSES)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Create Loader
    val_dataset = RADCUBE_DATASET_TIME(mode='test', params=params)

    # Create training and validation data loaders
    num_workers = 16
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=False)

    for batch in val_loader:

        radar_cube, lidar_cube, data_dict = batch

        with torch.no_grad():
            output = model(radar_cube)
            for i in range(lidar_cube.shape[0]):
                for t in range(lidar_cube.shape[1]):
                    output_t = output[i, t, :, :, :]
                    data_dict_t = data_dict[t]

                    radar_pc = data_preparation.cube_to_pointcloud(output_t, params, radar_cube[i, t, :, :, :], 'radar')

                    radar_pc[:, 2] = -radar_pc[:, 2]

                    cfar_path = data_dict_t["cfar_path"][i]
                    save_path = re.sub(r"radar_.+/", r"network/", cfar_path)
                    print(save_path)

                    np.save(save_path, radar_pc)


if __name__ == "__main__":
    params = data_preparation.get_default_params()

    # Initialise parameters
    params["dataset_path"] = "PATH_TO_DATASET"
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2, 6]
    params["train_test_split_percent"] = 0.8
    params["cfar_folder"] = 'radar_ososos'
    params["quantile"] = False

    # This must be kept to false. If the network without elevation is needed, use network_noElevation.py instead
    params["bev"] = False

    # This train the NN
    main(params)

    # This generate the poincloud from the trained NN
    # generate_point_clouds(params)

    # This compute the Pd, Pfa and Chamfer distance
    # compute_metrics(params)
