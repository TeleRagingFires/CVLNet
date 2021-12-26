# LCVNet
## Overview
This is a repository for LCVNet
The paper correspond to this code has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible. Here we will provide the test script and trained model to support reviewer to validate our work.



## Requirements
To use this project, you need to ensure the following requirements are installed.

- Python >= 3.7
- Tersorflow version we recommand 1.12.0
- Deep complex Network libaray: pip install keras-complex (supports TF1.13.0 at present)



## Project Preperation
#### A. Dataset

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA). The dataset path in our test script will be reserved for your local reproduction and validation.

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.


## Results and Reproduction

#### A. Overall result

## Acknowledgment

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Thanks Zhilin for his amazing work.
Thanks Chao-Kai Wen and Shi Jin group for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet) 
