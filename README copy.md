# Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory
### [Paper]()  | [Project Page]() 

> Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory

> Yuqi Wu<sup>\*</sup>, [Wenzhao Zheng](https://wzzheng.net/)<sup>*</sup>$\dagger$, [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)

<sup>*</sup> Equal contribution. $\dagger$ Project leader.

We propose Point3R, an online framework targeting **dense streaming 3D reconstruction**.

## Overview

![teaser](./assets/teaser.png)

We propose Point3R, an online framework targeting **dense streaming 3D reconstruction**.
Given streaming image inputs, our method maintains **an explicit spatial pointer memory** in which each pointer is assigned a 3D position and points to a changing spatial feature. We conduct a pointer-image interaction to integrate new observations into the global coordinate system and update our spatial pointer memory accordingly. Our method achieves competitive or state-of-the-art performance across various tasks: dense 3D reconstruction, monocular and video depth estimation, and camera pose estimation.

![overview](./assets/Main.png)

## Acknowledgements
Our code is based on the following awesome repositories:

- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r.git)
- [Spann3R](https://github.com/HengyiWang/spann3r.git)
- [CUT3R](https://github.com/CUT3R/CUT3R)

Very thanks to these authors!

## Citation

If you find this project helpful, please consider citing the following paper:
```

```

