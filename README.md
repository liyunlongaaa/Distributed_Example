# Distributed_Example

# 单机单卡

## python -m torch.distributed.launch --nproc_per_node=1 --use_env   main.py --epochs 10

# 单机多卡

## CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env   main.py --epochs 10
