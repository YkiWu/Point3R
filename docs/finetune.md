# Fine-tuning

If you want to fine-tune our checkpoint, you can use the following command.

You can modify the configuration file according to your own needs.

```

cd src/

# finetune 
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --num_processes=8 train.py  --config-name finetune

```
