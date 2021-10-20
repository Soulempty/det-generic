# coding: utf-8

import os 
import numpy as np
import argparse
from mmdet.apis import TrainInfo, Detection


#训练参数显示与设置

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def train(data_dir, img_size, batchsize=2, max_epochs=25):
    args = parse_args()

    # 训练前参数设置
    info = TrainInfo()
    info.set_param(data_dir, img_size=img_size, max_epochs=max_epochs, batchsize=batchsize)
    
    #训练实例初始化
    detect = Detection(gpu_id=1, distributed=args.distributed)
    #模型训练
    detect.train()


if __name__ == "__main__":
    data_dir = "data/train/bdd100"
    img_size = [1280,720]
    #train_cls = ['C4BP1', 'C4CK2', 'C4BP3', 'C4SH1', 'C4CP1', 'C4AR1', 'C4SP1', 'C4BP2', 'C4DP1']
    train(data_dir,img_size) 