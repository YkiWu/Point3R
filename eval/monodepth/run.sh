#!/bin/bash
set -e

workdir='/mnt/disk5/myspace/Point3R'
model_name='ours'
ckpt_name='stage3'
model_weights="/mnt/disk5/myspace/Point3R/src/checkpoints/stage3/checkpoint-final.pth"
datasets=('sintel' 'bonn' 'nyu' 'kitti')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth_512/${data}_${model_name}_${ckpt_name}"
    echo "$output_dir"
    python eval/monodepth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth_512/${data}_${model_name}_${ckpt_name}"
    python eval/monodepth/eval_metrics.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done

