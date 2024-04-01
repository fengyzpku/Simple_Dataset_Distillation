# I am now going to clean this code and pulish. Help me identify things that do not help and things that need comments.
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

import wandb

from framework.base import main_worker
from framework.config import get_arch

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Clean Train')
    parser.add_argument('--seed', default=0, type=int)
    # We have allowed distributed training on the data.
    parser.add_argument('--mp_distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='Number of processes')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')

    # Specify the GPU for one GPU training
    parser.add_argument('--gpu', default=None, type=int, help='GPU to use in non-distributed training')

    # Training configs
    parser.add_argument('--root', default='/scratch/yf2231/dataset', type=str, help='Root directory for dataset')
    parser.add_argument('--dataset', default='mnist', type=str, help='Dataset to use')
    parser.add_argument('--arch', default='mnistnet', type=str, help='Architecture to use')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for the distilled data')
    parser.add_argument('--inner_optim', default='SGD', type=str, help='Inner optimizer for the neural network')
    parser.add_argument('--outer_optim', default='Adam', type=str, help='Outer optimizer for the data')
    parser.add_argument('--inner_lr', default=0.01, type=float, help='inner learning rate')
    parser.add_argument('--label_lr_scale', default=1, type=float, help='scale the label lr')
    parser.add_argument('--num_per_class', default=1, type=int, help='Number of samples per class (IPC)')
    parser.add_argument('--batch_per_class', default=1, type=int, help='Number of samples per class per batch')
    parser.add_argument('--task_sampler_nc', default=10, type=int, help='Number of tasks to sample per batch')
    parser.add_argument('--window', default=20, type=int, help='Number of unrolling computing gradients')
    parser.add_argument('--minwindow', default=0, type=int, help='Start unrolling from steps x')
    parser.add_argument('--totwindow', default=20, type=int, help='Number of total unrolling computing gradients')
    parser.add_argument('--num_train_eval', default=10, type=int, help='Num of training of network for evaluation')
    parser.add_argument('--train_y', action='store_true', help='Train the label')
    parser.add_argument('--batch_size', default=200, type=int, help='Batch size for sampling from the original distribution')
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--test_freq', default=5, type=int, help='Frequency of testing in epochs')
    parser.add_argument('--print_freq', default=20, type=int, help='Frequency of printing in steps')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--ddtype', default='standard', type=str, help='Data Distillation Type, in curriculum, ')
    parser.add_argument('--cctype', default=0, type=int, help='Curriculum Type')
    # if cctype == 0: use a fix window without moving
    # if cctype == 1: use a window with moving forward
    # if cctype == 2: use a randomly placed window. The random location is changed every epoch
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--clip_coef', default=0.9, type=float, help='Clipping coefficient for the gradients in EMA')

    parser.add_argument('--fname', default='_test', type=str, help='Filename for storing checkpoints')
    parser.add_argument('--name', default='test', type=str, help='name of the experiment for wandb')
    parser.add_argument('--comp_aug', action='store_true', help='Compose different augmentation methods, if not, use only one randomly')
    parser.add_argument('--comp_aug_real', action='store_true', help='Compose different augmentation methods for the real data')
    parser.add_argument('--syn_strategy', default='flip_rotate', type=str, help='Synthetic data augmentation strategy')
    parser.add_argument('--real_strategy', default='flip_rotate', type=str, help='Real data augmentation strategy')
    parser.add_argument('--ckptname', default='none', type=str, help='Checkpoint name for initializing the distilled data')
    parser.add_argument('--limit_train', action='store_true', help='Limit the training data')
    parser.add_argument('--load_ckpt', action='store_true')
    parser.add_argument('--complete_random', action='store_true')
    
    
    args = parser.parse_args()

    args.distributed = args.world_size > 1 or args.mp_distributed
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
        
    args.num_train_eval = int(args.num_train_eval / ngpus_per_node)   
    
    if args.mp_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

        for i in range(5):
            torch.cuda.empty_cache()
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        
    
