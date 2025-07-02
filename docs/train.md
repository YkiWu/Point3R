# Training

We provide the following commands for training from scratch.
We train our model on a 8xH100 machine.

```
cd src/

# stage 1, 224 version + 5-frame sequences
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --num_processes=8 train.py  --config-name 224_stage1

# stage 2, 512 version + 5-frame sequences
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --num_processes=8 train.py  --config-name 512_stage2

# stage 3, freeze the encoder and fine-tune other parts on 8-frame sequences
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --num_processes=8 train.py  --config-name long_stage3
```