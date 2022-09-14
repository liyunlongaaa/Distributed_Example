# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print   #重写print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:  #python -m torch.distributed.launch  时为true
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, #init_method 内指定 tcp 模式，且所有进程的 ip:port 必须一致，设定为主进程的 ip:port
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier() #由于不同进程是同步执行的，单一进程处理数据必然会导致进程之间出现不同步的现象.如果执行create_dataloader()函数的进程不是主进程，即rank不等于0或者-1，上下文管理器会执行相应的torch.distributed.barrier()，设置一个阻塞栅栏，让此进程处于等待状态，等待所有进程到达栅栏处（包括主进程数据处理完毕）；如果执行create_dataloader()函数的进程是主进程，其会直接去读取数据并处理，然后其处理结束之后会接着遇到torch.distributed.barrier()，此时，所有进程都到达了当前的栅栏处，这样所有进程就达到了同步，并同时得到释放。
    setup_for_distributed(args.rank == 0)
