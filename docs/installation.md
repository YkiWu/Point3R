# Installation
Our code is based on the following environment.

## 1. Clone 
```bash
git clone https://github.com/YkiWu/Point3R.git
cd Point3R
```

## 2. Create conda environment
```bash
conda create -n point3r python=3.11 cmake=3.14.0
conda activate point3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install -r requirements.txt
conda install 'llvm-openmp<16'
```
