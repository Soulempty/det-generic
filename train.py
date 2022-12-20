# coding: utf-8

import os 
import numpy as np
import argparse
from mmdet.apis import Detection
from mmcv.runner import get_dist_info, init_dist

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--train_cls', nargs='+', help='the defect code to train!',default=[])
    parser.add_argument('--save_dir', type=str, default='std-2')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def train(args, data_dir, img_size, train_cls=[], work_dir=None, gpu_ids=[0], batchsize=2, max_epochs=25):
 
    if args.distributed:
        init_dist('pytorch', backend="nccl")

    print(f"Distributed training:{args.distributed}")

    # 训练前参数设置
    detect = Detection(gpu_ids, distributed=args.distributed)
    detect.set_param(data_dir, train_cls,img_size=img_size, max_epochs=max_epochs, batchsize=batchsize,work_dir=work_dir)

    #模型训练ss
    detect.train()


if __name__ == "__main__":
    args = parse_args()
    data_dir = "data/wxn"
    img_size = [1560,1280]
    train_cls = [] #args.train_cls
    train(args,data_dir,img_size,train_cls)  
