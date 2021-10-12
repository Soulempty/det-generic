#coding:utf-8

import argparse
import copy
import os
import time
import json
import random

import mmcv
import torch
import numpy as np
import logging
from mmcv import Config, DictAction

from .utils import replace_ImageToTensor,data_split,async_,get_train_cls

from mmdet import __version__
from mmdet.datasets import build_dataset,build_dataloader
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.core import DistEvalHook, EvalHook, Fp16OptimizerHook
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook, build_optimizer, init_dist)

root_dir = os.path.join(os.path.dirname(__file__),"../..")
data_cfg_path = os.path.join(root_dir,"configs/_base_/datasets/train_data.json")
model_cfg_path = os.path.join(root_dir,"configs/_base_/models/model.json")
test_cfg_path = os.path.join(root_dir,"configs/_base_/cfgs/test_cfg.json")
train_cfg_path = os.path.join(root_dir,"configs/_base_/cfgs/train_cfg.json")
runtime_cfg_path = os.path.join(root_dir,"configs/_base_/cfgs/runtime_cfg.json")

class TrainInfo():
    def __init__(self): #

        self.data_cfg = mmcv.Config.fromfile(data_cfg_path) 
        self.model_cfg = mmcv.Config.fromfile(model_cfg_path)
        self.runtime_cfg = mmcv.Config.fromfile(runtime_cfg_path)
        self.data_path = None
    
    def split_data(self,lb_dir,img_dir):
        data_split(self.data_path,lb_dir,img_dir)
        return get_train_cls(self.data_path,lb_dir)

    def set_lr(self,learning_rate=None):
        if learning_rate:
            self.runtime_cfg['runtime_cfg']["optimizer"]["lr"] = learning_rate 
            self.runtime_cfg.dump(runtime_cfg_path)
        return 0


    def set_param(self,data_dir,step_id,train_cls=None,max_epochs=25,batchsize=2,learning_rate=0.016,work_dir=None,img_dir='source',label_dir='label'):
        if batchsize:
            self.data_cfg['data']['samples_per_gpu'] = batchsize
        if os.path.isabs(data_dir):
            self.data_cfg['data']['train']['data_root'] = data_dir+'/'
        else:
            self.data_cfg['data']['train']['data_root'] = os.path.join(self.data_cfg['data']['data_path'],data_dir)+'/'
        self.data_path = self.data_cfg['data']['train']['data_root']
        self.data_cfg['data']['train']['img_prefix'] = self.data_path
        self.data_cfg['data']['val']['data_root'] = self.data_path
        self.data_cfg['data']['val']['img_prefix'] = self.data_path
        self.data_cfg['data']['step_id'] = step_id

        train_codes = self.split_data(label_dir,img_dir)
        if train_cls:
            train_codes = train_cls
        if learning_rate:
            self.runtime_cfg['runtime_cfg']["optimizer"]["lr"] = learning_rate 

        self.data_cfg['data']['train']['code_cls'] = train_codes
        self.data_cfg['data']['val']['code_cls'] = train_codes
        self.model_cfg['model']['roi_head']['bbox_head']['num_classes'] = len(train_codes)

        self.runtime_cfg['runtime_cfg']['work_dir'] = self.data_cfg['data']['train']['data_root']+'model'
        
        if work_dir and os.path.exists(work_dir):
            self.runtime_cfg['runtime_cfg']['work_dir'] = work_dir
        if max_epochs:
            self.runtime_cfg['runtime_cfg']['total_epochs'] = max_epochs
            step = int(round((max_epochs-max_epochs//7)/3))                      
            self.runtime_cfg['runtime_cfg']['lr_config']['step'] = [step*2,step*3]
        if img_dir:
            self.data_cfg['data']['train']['img_dir'] = img_dir
        if label_dir:
            self.data_cfg['data']['train']['label_dir'] = label_dir
        self.data_cfg.dump(data_cfg_path)
        self.model_cfg.dump(model_cfg_path)
        self.runtime_cfg.dump(runtime_cfg_path)



# gpu_id
# resume: 
# load:        
# validate
# distributed

class Detection():
    def __init__(self,gpu_id=0,validate=False,distributed=False,resume_from=None,load_from=None,test_score=0.2):
        super().__init__()
        self.data_cfg = mmcv.Config.fromfile(data_cfg_path)
        self.model_cfg = mmcv.Config.fromfile(model_cfg_path)
        self.train_cfg = mmcv.Config.fromfile(train_cfg_path)
        self.test_cfg = mmcv.Config.fromfile(test_cfg_path)
        self.test_cfg.test_cfg["rcnn"]["score_thr"] = test_score
        self.runtime_cfg = mmcv.Config.fromfile(runtime_cfg_path)
        self.resume_from = resume_from
        self.load_from = load_from
        self.distributed = distributed
        self.validate = validate
        self.gpu_ids = [gpu_id,]
        self.train_cls = []
        self.meta = dict()
        self.model = None
        self.optimizer = None
        self._max_iters = 0
        self._max_epochs = self.runtime_cfg.runtime_cfg.total_epochs
        self._data_path = self.data_cfg['data']['train']['data_root']
        self._stop_flag = False
        self._model_path = None
        self._task_flag = self.data_path.split('/')[-2] if self.data_path.split('/')[-1]=='' else self.data_path.split('/')[-1]

        self.model_init_flag = False
        self.data_init_flag = False
        

        self.work_dir = os.path.abspath(self.runtime_cfg.runtime_cfg.work_dir)
        mmcv.mkdir_or_exist(self.work_dir)

        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(self.work_dir, f'{self.timestamp}.log')
        self.dash_line = '-' * 60 + '\n'
        self.logger = get_root_logger(log_file=log_file, log_level=self.runtime_cfg.runtime_cfg.log_level)
        self.logger.info("Adc Trainer Info:\n")

        self.data_cfg.dump(os.path.join(self.work_dir, os.path.basename(data_cfg_path)))
        self.model_cfg.dump(os.path.join(self.work_dir, os.path.basename(model_cfg_path)))
        self.runtime_cfg.dump(os.path.join(self.work_dir, os.path.basename(runtime_cfg_path)))
        self.train_cfg.dump(os.path.join(self.work_dir, os.path.basename(train_cfg_path)))

        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        self.meta['env_info'] = env_info
        self.data_loaders = None
        self.init_dataset()
        
    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def task_flag(self):
        return self._task_flag

    @property
    def model_path(self):
        return self._model_path

    def update_model_path(self,epoch_):
        filename_tmpl = 'model_'+self.task_flag+"_{}.pth"
        self._model_path = os.path.join(self.data_path,'model',filename_tmpl.format(epoch_))  
    @property
    def data_path(self):
        return self._data_path

    @property
    def max_iters(self):
        return self._max_iters
    
    def release(self):
        if self.model_init_flag:
            self.model = self.model.cpu()
            del self.model
            del self.optimizer
        if self.data_init_flag:
            del self.data_loaders

        torch.cuda.empty_cache() 
        torch.cuda.empty_cache() 
        time.sleep(2)
        return

    def init_model(self):
        try:
            model = build_detector(self.model_cfg.model, train_cfg=self.train_cfg.train_cfg, test_cfg=self.test_cfg.test_cfg)
            model.CLASSES = self.train_cls
            if not self.distributed:
                self.model = MMDataParallel(model.cuda(self.gpu_ids[0]), device_ids=self.gpu_ids)
            else:
                self.model = MMDistributedDataParallel(model.cuda(),device_ids=[torch.cuda.current_device()],broadcast_buffers=False,find_unused_parameters=False)
            self.optimizer = build_optimizer(self.model, self.runtime_cfg.runtime_cfg.optimizer)
            self.logger.info("build detector and optimizer successfully!\n")
        except Exception as e:
            self.logger.error("build detector and optimizer failed!\n")
        self.model_init_flag = True

    def init_dataset(self):
        datasets = [build_dataset(self.data_cfg.data.train)]
        self.train_cls = datasets[0].CLASSES
        try:
            data_loaders = [build_dataloader(
                    ds,
                    self.data_cfg.data.samples_per_gpu,
                    self.data_cfg.data.workers_per_gpu,
                    len(self.gpu_ids),
                    dist=self.distributed,
                    seed=None) for ds in datasets]
            self.logger.info("train dataloader loading successfully!\n")
        except Exception as e:
            self.logger.error("train dataloader loading failed!\n")

        self._max_iters = self._max_epochs * len(data_loaders[0])
        self.runtime_cfg.runtime_cfg.lr_config.warmup_iters = len(data_loaders[0])*2
        # add some important info to meta of model tobe saved.
        if self.runtime_cfg.runtime_cfg.checkpoint_config is not None:
            self.runtime_cfg.runtime_cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__,
                CLASSES=self.train_cls,
                step_id=self.data_cfg.data.step_id)
        self.data_loaders = data_loaders
        self.data_init_flag = True
    
    def stop(self):
        self._stop_flag = True
    
    def stop_flag(self):
        return self._stop_flag
    
    @async_
    def train(self,call_back=None):
        self.logger.info('Environment info:\n' + self.dash_line + self.meta['env_info'] + '\n' + self.dash_line) 

        if self.distributed:
            init_dist('pytorch', **self.runtime_cfg.runtime_cfg.dist_params)
        #datasets, data_loaders = self.init_dataset()
        self.init_model()
        
        runner = EpochBasedRunner(
            self.model,
            optimizer=self.optimizer,
            work_dir=self.work_dir,
            logger=self.logger,
            meta=self.meta,
            task_flag=self.task_flag,
            stop_flag=self.stop_flag,
            call_back=call_back,
            update_model_path=self.update_model_path
            )
        runner.timestamp = self.timestamp
        # fp16 setting
        fp16_cfg = self.runtime_cfg.runtime_cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(**self.runtime_cfg.runtime_cfg.optimizer_config, **fp16_cfg, distributed=self.distributed)
        else:
            optimizer_config = self.runtime_cfg.runtime_cfg.optimizer_config
        try:
            runner.register_training_hooks(self.runtime_cfg.runtime_cfg.lr_config, optimizer_config,
                                           self.runtime_cfg.runtime_cfg.checkpoint_config, 
                                           self.runtime_cfg.runtime_cfg.log_config,
                                           None)
            self.logger.info("register training hooks successfully!\n")
        except Exception as e:
            self.logger.error("register training hooks failed!\n")

        if self.validate:
            samples_per_gpu = self.data_cfg.data.val.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                self.data_cfg.data.val.pipeline = replace_ImageToTensor(self.data_cfg.data.val.pipeline)
            val_dataset = build_dataset(self.data_cfg.data.val, dict(test_mode=True))
            try:
                val_dataloader = build_dataloader(
                    val_dataset,
                    samples_per_gpu=samples_per_gpu,
                    workers_per_gpu=self.data_cfg.data.workers_per_gpu,
                    dist=self.distributed,
                    shuffle=False)
                self.logger.info("test dataloader loading successfully!\n")
            except Exception as e:
                self.logger.error("test dataloader loading failed!\n")

            eval_cfg = self.runtime_cfg.runtime_cfg.get('evaluation', {})
            eval_hook = DistEvalHook if self.distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
        if self.resume_from:
            runner.resume(self.resume_from)
        elif self.load_from:
            runner.load_checkpoint(self.load_from)
        runner.run(self.data_loaders, self.runtime_cfg.runtime_cfg.workflow, self.runtime_cfg.runtime_cfg.total_epochs)
        

if __name__ == '__main__':
    main()


