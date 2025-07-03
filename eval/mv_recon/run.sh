#!/bin/bash

set -e

workdir='/mnt/disk5/myspace/Point3R'
model_name='ours'
ckpt_name='stage3'
model_weights="/mnt/disk5/myspace/Point3R/src/checkpoints/point3r_512.pth"

output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}"
echo "$output_dir"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --num_processes 8 --main_process_port 29998 eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --size 512
