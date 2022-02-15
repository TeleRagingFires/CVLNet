# Q_Uni_CRNet
## Overview
This is a repository for Reporducing [CRNet](https://ieeexplore.ieee.org/document/9149229) with End-to-End Uniform Quantization and Dequantization.

## Requirements
To use this project, you need to ensure the following requirements are installed.

- Python >= 3.7
- Tersorflow version we recommand 1.12.0


## Project Preperation
#### A. Dataset

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA). The dataset path in our test script will be reserved for your local reproduction and validation.

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.


## Results and Reproduction
The result will be printed by the test script, in the absence of the confirmation from the author, 


## Acknowledgment

The very creative Complex-valued network components can be found here [Keras-complex](https://github.com/JesperDramsch/keras-complex). This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Thanks Zhilin for his amazing work.
Thanks Chao-Kai Wen and Shi Jin group for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet). 

