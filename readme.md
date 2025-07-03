# EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding
### [Paper](https://arxiv.org/abs/2412.04380)  | [Project Page](https://ykiwu.github.io/EmbodiedOcc/)


> EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding

> Yuqi Wu, [Wenzhao Zheng](https://wzzheng.net/)$\dagger$, [Sicheng Zuo](https://scholar.google.com/citations?user=11kh6C4AAAAJ&hl=en&oi=ao), [Yuanhui Huang](https://scholar.google.com/citations?hl=zh-CN&user=LKVgsk4AAAAJ), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)

$\dagger$ Project leader

EmbodiedOcc formulates **an embodied 3D occupancy prediction task** and proposes a Gaussian-based framework to accomplish it.

![teaser](./img/teaser_v4.png)

## Overview

Targeting progressive embodied exploration in indoor scenarios, we formulate an embodied 3D occupancy prediction task and propose a Gaussian-based EmbodiedOcc framework accordingly.
Our EmbodiedOcc maintains an explicit Gaussian memory of the current scene and updates this memory during the exploration of this scene.
Both quantitative and visualization results have shown that our EmbodiedOcc outperforms existing methods in terms of local occupancy prediction and accomplishes the embodied occupancy prediction task with high accuracy and strong expandability.

![overview](./img/Main.png)

## Related Projects

Our work is inspired by these excellent open-sourced repos:
[GaussianFormer](https://github.com/huang-yh/GaussianFormer)
[ISO](https://github.com/hongxiaoy/ISO)

Our code is based on [GaussianFormer](https://github.com/huang-yh/GaussianFormer).

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{wu2024embodiedoccembodied3doccupancy,
      title={EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding}, 
      author={Yuqi Wu and Wenzhao Zheng and Sicheng Zuo and Yuanhui Huang and Jie Zhou and Jiwen Lu},
      journal={arXiv preprint arXiv:2412.04380},
      year={2024}
}
```
