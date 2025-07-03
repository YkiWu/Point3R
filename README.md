# Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory
### [Paper]()  | [Project Page]() 

> Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory

> Yuqi Wu<sup>\*</sup>, [Wenzhao Zheng](https://wzzheng.net/)<sup>*</sup>$\dagger$, [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)

<sup>*</sup> Equal contribution. $\dagger$ Project leader.

We propose **Point3R**, an online framework targeting **dense streaming 3D reconstruction** using explicit spatial memory.

## Overview

<img src="./assets/teaser.png" alt="overview" style="width: 100%;" />

Given streaming image inputs, our method maintains **an explicit spatial pointer memory** in which each pointer is assigned a 3D position and points to a changing spatial feature. We conduct a pointer-image interaction to integrate new observations into the global coordinate system and update our spatial pointer memory accordingly. Our method achieves competitive or state-of-the-art performance across various tasks: dense 3D reconstruction, monocular and video depth estimation, and camera pose estimation.

<img src="./assets/Main.png" alt="overview" style="width: 100%;" />

## Getting Started

### Installation
Our code is based on the following environment.

#### 1. Clone 
```bash
git clone https://github.com/YkiWu/Point3R.git
cd Point3R
```

#### 2. Create conda environment
```bash
conda create -n point3r python=3.11 cmake=3.14.0
conda activate point3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install -r requirements.txt
conda install 'llvm-openmp<16'
```

### Data Preparation
Please follow [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) to prepare the training datasets. The official links of all used datasets are listed below.

  - [ARKitScenes](https://github.com/apple/ARKitScenes) 
  - [BlendedMVS](https://github.com/YoYo000/BlendedMVS)
  - [CO3Dv2](https://github.com/facebookresearch/co3d)
  - [Hypersim](https://github.com/apple/ml-hypersim)
  - [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/)
  - [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html)
  - [OmniObject3D](https://omniobject3d.github.io/)
  - [PointOdyssey](https://pointodyssey.com/)
  - [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) 
  - [ScanNet](http://www.scan-net.org/ScanNet/)
  - [Spring](https://spring-benchmark.org/)
  - [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
  - [WayMo Open dataset](https://github.com/waymo-research/waymo-open-dataset)
  - [WildRGB-D](https://github.com/wildrgbd/wildrgbd/)

## Training from Scratch

We provide the following commands for training from scratch.

Please download [`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) and place it on your own path.

```
cd src/

# stage 1, 224 version + 5-frame sequences
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --num_processes=8 train.py  --config-name 224_stage1

# stage 2, 512 version + 5-frame sequences
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --num_processes=8 train.py  --config-name 512_stage2

# stage 3, freeze the encoder and fine-tune other parts on 8-frame sequences
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --num_processes=8 train.py  --config-name long_stage3
```

## Fine-tuning

If you want to fine-tune our checkpoint, you can use the following command.

#### 1. Download Checkpoints
Click [HERE]() to download our checkpoint and place it on your own path.

#### 2. Start Finetuning
You can modify the configuration file according to your own needs.

```
cd src/

# finetune 
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --num_processes=8 train.py  --config-name finetune

```

## Evaluation

### Data Preparation
Please follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare the evaluation datasets.

### Scripts

Our evaluation code follows [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/eval.md).

#### 3D Reconstruction

```bash
bash eval/mv_recon/run.sh
```

Results will be saved in `eval_results/mv_recon/${model_name}_${ckpt_name}/logs_all.txt`.

#### Monodepth

```bash
bash eval/monodepth/run.sh
```
Results will be saved in `eval_results/monodepth/${data}_${model_name}/metric.json`.

#### Video Depth

```bash
bash eval/video_depth/run.sh 
```
Results will be saved in `eval_results/video_depth/${data}_${model_name}/result_scale.json`.

#### Camera Pose Estimation

```bash
bash eval/relpose/run.sh 
```
Results will be saved in `eval_results/relpose/${data}_${model_name}/_error_log.txt`.


## Acknowledgements
Our code is based on the following awesome repositories:

- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r.git)
- [Spann3R](https://github.com/HengyiWang/spann3r.git)
- [CUT3R](https://github.com/CUT3R/CUT3R)

Many thanks to these authors!

## Citation

If you find this project helpful, please consider citing the following paper:
```

```

