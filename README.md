<p align="center">
  <img src = "docs/figures/logo.png" width="60%">
</p>

# RaDelft Dataset

This repository shares the documentation for the RaDelf dataset as well as the code for reproducing the results of [1].

<div align="center">
<p float="center">
<img src="docs/figures/exampleVideo.gif" alt="Example video" width="600"/>
<br />
<b>Example video from our dataset, with the camera on top, lidar on the right and the point cloud from [1] on the left.</b>
</p>
</div>

## Overview
- [Introduction](#introduction)
- [Sensors and Data](#sensors-and-data)

- [Access](#access)
- [Getting Started](#getting-started)
- [Examples and Demo](#examples-and-demos)
- [Annotation](#annotation)
- [License] (#license)
- [Citation](#citation)
- [Original paper](Coming Soon)
- [Links](#links)


## Changelog

## Introduction

The RaDelft dataset is a large-scale, real-life multi-sensor dataset recorded in various driving scenarios. It provides radar data in different processing levels, synchronised with lidar, camera and odometry.

## Sensors and data
The output of the next sensors have been recorded:

- A texas instruments MIMO radar board MMWCAS-RF-EVM mounted on the roof.
- A RoboSense Ruby Plus Lidar (128 layers rotating lidar) mounted on the roof.
- A video camera mounted on the windshield (1936 x 1216 px, ~30Hz).
- The ego vehicle’s odometry (filtered combination of RTK GPS, IMU, and wheel odometry, ∼100 Hz).

All sensors were jointly calibrated. See the figure below for a general overview of the sensor setup.

<div align="center">
<p float="center">
<img src="docs/figures/car2.png" alt="Example video" width="300"/>
</p>
</div>

## Access
The dataset is made freely available for non-commercial research purposes only. Eligibility to use the dataset is limited to Master- and PhD-students, and staff of academic and non-profit research institutions. The dataset is hosted in 4TU.ResearchData:

> [!NOTE]  
> Link coming soon.

By requesting access, the researcher agrees to use and handle the data according to the license.

After validating the researcher’s association to a research institue, we will send an email containing password protected download link(s) of the RaDelft dataset. Sharing these links and/or the passwords is strictly forbidden (see licence).

In case of questions of problems, please send an email to i.roldanmontero at tudelft.nl.


Frequently asked questions about the license:

Q: Is it possible for MSc and PhD students to use the dataset if they work/cooperate with a for-profit organization?
A: The current VoD license permits the use of the VoD dataset of a MS/PhD student on the compute facilities (storing, processing) of his/her academic institution for research towards his/her degree - even if this MS/PhD student is (also) employed by a company.
The license does not permit the use of the VoD dataset on the compute facilities (storing, processing) of a for-profit organization.

## Getting Started
Coming soon
## Examples and Demos
---
### 1_frame_loader - Frame Loader
This example shows how to generate the dataloader for the single and multiframe versions.

[Link To the Jupyter Notebook](https://github.com/IgnacioRoldan/RaDelft-Dataset/blob/main/machine_learning_python/examples/1_frame_loader.ipynb)

### 2_3d_visualization - 3D Visualization
This example notebook shows how to load the plot the 3D point clouds of the lidar and radar with the NN detector.

[Link To the Jupyter Notebook](https://github.com/IgnacioRoldan/RaDelft-Dataset/blob/main/machine_learning_python/examples/2_3DVisualiser.ipynb)

### 3_camera_projection - Point clouds projected on camera
---


## Annotation
> [!NOTE]  
> Labeling of the data is being made to enable classification and segmentation algorithms. Labels will be released in the following updates.

## License
* The development kit is realeased under the Apache License, Version 2.0, see [here](LICENSE.txt).
* The dataset can be used by accepting the [Research Use License](https://data.4tu.nl).

## Citation
Coming soon
## Links
Coming soon
