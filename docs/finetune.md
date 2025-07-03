# Fine-tuning

If you want to fine-tune our checkpoint to handle longer sequences or using your own training datasets, you can use the following command.

```

cd src/

# finetune 
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --num_processes=8 train.py  --config-name finetune

```