#!/bin/bash

set -e

workdir='/mnt/disk5/myspace/Point3R'
model_name='ours'
ckpt_name='stage3'
model_weights="/mnt/disk5/myspace/Point3R/src/checkpoints/stage3/checkpoint-final.pth"
datasets=('sintel' 'tum' 'scannet')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/relpose_512/${data}_${model_name}_${ckpt_name}"
    echo "$output_dir"
    accelerate launch --num_processes 8 --main_process_port 29558 eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512
done


