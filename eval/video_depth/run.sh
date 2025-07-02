#!/bin/bash

set -e

workdir='/mnt/disk5/myspace/Point3R'
model_name='ours'
ckpt_name='stage3'
model_weights="/mnt/disk5/myspace/Point3R/src/checkpoints/stage3/checkpoint-final.pth"
datasets=('sintel' 'bonn' 'kitti')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}_${ckpt_name}"
    echo "$output_dir"
    accelerate launch --num_processes 8  eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512
    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale&shift"
done
