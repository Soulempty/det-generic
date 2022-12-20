#coding:utf-8

import argparse
import copy
import os
import sys 
import time
import json
import random

import mmcv
import torch
import traceback
import numpy as np
import logging
from mmcv import Config, DictAction

from .utils import replace_ImageToTensor,data_split,async_,get_train_cls,get_logger,logger_initialized, torch_distributed_zero_first

from mmdet import __version__
from mmdet.datasets import build_dataset,build_dataloader
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.core import DistEvalHook, EvalHook, Fp16OptimizerHook
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook, build_optimizer, init_dist, get_dist_info

root_dir = os.path.join(os.path.dirname(__file__),"../..")
data_cfg_path = os.path.join(root_dir,"configs/_base_/datasets/train_data.json")
model_cfg_path = os.path.join(root_dir,"configs/_base_/models/model.json")
test_cfg_path = os.path.join(root_dir,"configs/_base_/cfgs/test_cfg.json")
train_cfg_path = os.path.join(root_dir,"configs/_base_/cfgs/train_cfg.json")
runtime_cfg_path = os.path.join(root_dir,"configs/_base_/cfgs/runtime_cfg.json")


class Detection():
    def __init__(self,gpu_ids=[0,],validate=False,distributed=False,resume_from=None,load_from=None,test_score=0.2):
        super().__init__()
        self.data_cfg = mmcv.Config.fromfile(data_cfg_path)
        self.model_cfg = mmcv.Config.fromfile(model_cfg_path)
        self.train_cfg = mmcv.Config.fromfile(train_cfg_path)
        self.test_cfg = mmcv.Config.fromfile(test_cfg_path)
        self.runtime_cfg = mmcv.Config.fromfile(runtime_cfg_path)
        if "rcnn" in self.test_cfg.test_cfg:
            self.test_cfg.test_cfg["rcnn"]["score_thr"] = test_score
        self.gpu_ids = gpu_ids
        self.validate = validate
        self.distributed = distributed
        self.resume_from = resume_from
        self.load_from = load_from
        
        self.train_cls = []
        self.meta = dict()
        self.model = None
        self.optimizer = None
        self._max_iters = 0
        self._max_epochs = 25
        self._stop_flag = False
        self._task_flag = 'dummy'
        self._model_path = None
        self.work_dir = ''
        self.model_init_flag = False
        self.data_init_flag = False
        self.flag = 0
        self.logger = None
        self._data_path = None

        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        self.meta['env_info'] = env_info
        self.data_loaders = None

    def set_param(self,data_dir,train_cls=[],img_size=None,max_epochs=25,batchsize=4,learning_rate=0.01,work_dir=None,img_dir='source',label_dir='label'):
        
        data_dir = os.path.abspath(data_dir)
        if os.path.exists(data_dir):
            self._data_path = data_dir+'/'
        else:
            self._data_path = os.path.join(self.data_cfg['data']['data_path'],data_dir)+'/'

        self.data_cfg['data']['train']['data_root'] = self.data_path
        self.data_cfg['data']['val']['data_root'] = self.data_path
        
        train_codes = self.split_data(label_dir,img_dir,train_cls)
        if train_cls:
            train_codes = train_cls
        self.data_cfg['data']['train']['code_cls'] = train_codes
        self.data_cfg['data']['val']['code_cls'] = train_codes
        if 'roi_head' in self.model_cfg['model']:
            self.model_cfg['model']['roi_head']['bbox_head']['num_classes'] = len(train_codes)

        if max_epochs:
            self._max_epochs = max_epochs
            self.runtime_cfg['runtime_cfg']['total_epochs'] = max(max_epochs,5)
            step = int(round((max_epochs-max_epochs//7)/3))                      
            self.runtime_cfg['runtime_cfg']['lr_config']['step'] = [step*2,step*3]

        if batchsize:
            self.data_cfg['data']['samples_per_gpu'] = batchsize

        if learning_rate:
            self.runtime_cfg['runtime_cfg']["optimizer"]["lr"] = learning_rate 

        self.work_dir = os.path.abspath(self.data_path + 'model')
        if work_dir and os.path.exists(work_dir):
            self.work_dir = work_dir

        mmcv.mkdir_or_exist(self.work_dir)
        self.runtime_cfg['runtime_cfg']['work_dir'] = self.work_dir
        
        if img_dir:
            self.data_cfg['data']['train']['img_dir'] = img_dir
        if label_dir:
            self.data_cfg['data']['train']['label_dir'] = label_dir

        if self.data_path.split('/')[-1]=='':
            self._task_flag = self.data_path.split('/')[-2] 
        else:
            self._task_flag = self.data_path.split('/')[-1]

        if img_size:
            self.data_cfg["data"]["train"]["pipeline"][2]["img_scale"] = img_size # w,h
            self.data_cfg["data"]["val"]["pipeline"][1]["img_scale"] = img_size

        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_path = os.path.join(self.data_cfg['data']['log_path'])
        mmcv.mkdir_or_exist(log_path)

        log_file = os.path.join(log_path, f'{self.timestamp}.log')
        gpu_id = self.gpu_ids[0]
        self.log_name = f"Aqrose[{self.task_flag}-{gpu_id}]"
        self.logger = get_logger(name=self.log_name,log_file=log_file, log_level=self.runtime_cfg.runtime_cfg.log_level)
        self.logger.info("******  Aqrose's version: AI For Every Factory! ******")
        self.logger.info(f"The workdir path of current training task is:{self.work_dir}.")

        self.data_cfg.dump(os.path.join(self.work_dir, os.path.basename(data_cfg_path)))
        self.model_cfg.dump(os.path.join(self.work_dir, os.path.basename(model_cfg_path)))
        self.runtime_cfg.dump(os.path.join(self.work_dir, os.path.basename(runtime_cfg_path)))
        self.train_cfg.dump(os.path.join(self.work_dir, os.path.basename(train_cfg_path)))
        self.init_dataset()
        return 
    
    def split_data(self,lb_dir,img_dir,codes):
        # rank, world_size = get_dist_info()
        # with torch_distributed_zero_first(rank):
        data_split(self.data_path,lb_dir,img_dir,codes)
        return get_train_cls(self.data_path,lb_dir)

    def set_lr(self,learning_rate=None):
        if learning_rate:
            self.runtime_cfg['runtime_cfg']["optimizer"]["lr"] = learning_rate 
        return 0

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
        self._model_path = os.path.join(self.work_dir,filename_tmpl.format(epoch_))  

    @property
    def data_path(self):
        return self._data_path

    @property
    def max_iters(self):
        return self._max_iters
    
    @staticmethod
    def gpu_info(gpu_id=None):
        if isinstance(gpu_id,int):
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id) 
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = meminfo.used/1024**2  
            free = meminfo.free/1024**2 
            return (used,free)
        else:
            return (0,0)

    def release(self):
        try:
            if self.model_init_flag:
                self.model = self.model.cpu()
                del self.optimizer
                del self.model
            self.logger.info("After finishing training,the optimizer and model are released!")
        except Exception as e:
            self.logger.error(f"After finishing training,{e} occured while the optimizer and model are released!")
            traceback.print_exc()
        try:
            if self.data_init_flag:
                del self.data_loaders
            self.logger.info("After finishing training,the dataloaders are released!")
        except Exception as e:
            self.logger.info(f"After finishing training,{e} occured while the dataloaders are released!")
            traceback.print_exc()
        
        if self.log_name in logger_initialized:
            self.logger.info(f"After finishing training, delete the logger var of name {self.log_name}!")
            del logger_initialized[self.log_name]
            
        torch.cuda.empty_cache() 
        torch.cuda.empty_cache() 
        time.sleep(2)
        return

    def init_model(self):

        self.model_init_flag = True
        try:
            model = build_detector(self.model_cfg.model, train_cfg=self.train_cfg.train_cfg, test_cfg=self.test_cfg.test_cfg)
            model.CLASSES = self.train_cls
            self.logger.info("build detector successfully!")
        except Exception as e:
            self.logger.error(f"build detector failed, and error is {e}.")
            traceback.print_exc()
            self.model_init_flag = False
        try:
            if not self.distributed:
                self.model = MMDataParallel(model.cuda(self.gpu_ids[0]), device_ids=self.gpu_ids)
            else:
                self.model = MMDistributedDataParallel(model.cuda(),device_ids=[torch.cuda.current_device()],broadcast_buffers=False,find_unused_parameters=False)
            self.logger.info("paralleling the model successfully!")
        except Exception as e:
            self.logger.error(f"error {e} occurred while paralleling the model!")
            traceback.print_exc()
            self.model_init_flag = False
        try:
            self.optimizer = build_optimizer(self.model, self.runtime_cfg.runtime_cfg.optimizer)
            self.logger.info("build optimizer successfully!")
        except Exception as e:
            self.logger.error(f"build optimizer failed,error is {e}.")
            traceback.print_exc()
            self.model_init_flag = False

    def init_dataset(self):
        datasets = []
        data_loaders = []
        self.data_init_flag = True

        try:
            datasets = [build_dataset(self.data_cfg.data.train)]
            self.train_cls = datasets[0].CLASSES
            self.logger.info("build dataset successfully!")
        except Exception as e:
            self.logger.error(f"build dataset failed,error is {e}.")
            traceback.print_exc()
            self.data_init_flag = False
        try:
            data_loaders = [build_dataloader(
                    ds,
                    self.data_cfg.data.samples_per_gpu,
                    self.data_cfg.data.workers_per_gpu,
                    len(self.gpu_ids),
                    dist=self.distributed,
                    seed=None) for ds in datasets]
            self.logger.info("train dataloader buildding successfully!")

            self._max_iters = self.max_epochs * len(data_loaders[0])
            self.logger.info(f"The max iters of the training model is {self.max_iters}.")
            self.runtime_cfg.runtime_cfg.lr_config.warmup_iters = len(data_loaders[0])*2

            # add some important info to meta of model tobe saved.
            if self.runtime_cfg.runtime_cfg.checkpoint_config is not None:
                self.runtime_cfg.runtime_cfg.checkpoint_config.meta = dict(
                    mmdet_version=__version__,
                    CLASSES=self.train_cls)
            self.data_loaders = data_loaders

        except Exception as e:
            self.logger.error(f"train dataloader buildding failed,error is {e}.")
            traceback.print_exc()
            self.data_init_flag = False
   
    def set_flag(self):
        self._stop_flag = True
        return 0
    
    def stop(self):
        self.set_flag()
        while True:
            if self.flag == -2:
                break
    
    def stop_flag(self):
        return self._stop_flag
    
    # @async_
    def train(self):

        dash_line = '-' * 60 + '\n'
        self.logger.info('ADC Traing Environment info:\n' + dash_line + self.meta['env_info'] + '\n' + dash_line) 

        if not self.data_init_flag:
            self.logger.error("The data processing encountered error!\n")

        self.init_model()
        if not self.model_init_flag:
            self.logger.error("The model initializing encountered error!\n")

        runner = None
        try:
            runner = EpochBasedRunner(
                self.model,
                optimizer=self.optimizer,
                work_dir=self.work_dir,
                logger=self.logger,
                meta=self.meta
                )
        except Exception as e:
            self.logger.error(f"runner's initialization fron EpochBasedRunner is failed,error is {e}.")
            traceback.print_exc()

        runner.timestamp = self.timestamp
        # fp16 setting
        fp16_cfg = self.runtime_cfg.runtime_cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(**self.runtime_cfg.runtime_cfg.optimizer_config, **fp16_cfg, distributed=self.distributed)
        elif self.distributed and 'type' not in self.runtime_cfg.runtime_cfg.optimizer_config:
            optimizer_config = OptimizerHook(**self.runtime_cfg.runtime_cfg.optimizer_config)
        else:
            optimizer_config = self.runtime_cfg.runtime_cfg.optimizer_config
        try:
            runner.register_training_hooks(self.runtime_cfg.runtime_cfg.lr_config, optimizer_config,
                                           self.runtime_cfg.runtime_cfg.checkpoint_config, 
                                           self.runtime_cfg.runtime_cfg.log_config,
                                           None)
            self.logger.info("register training hooks successfully!")
        except Exception as e:
            self.logger.error(f"register training hooks failed,error is {e}.")
            traceback.print_exc()

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
                self.logger.info("test dataloader loading successfully!")
            except Exception as e:
                self.logger.error("test dataloader loading failed!")
                traceback.print_exc()

            eval_cfg = self.runtime_cfg.runtime_cfg.get('evaluation', {})
            eval_hook = DistEvalHook if self.distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
        if self.resume_from:
            runner.resume(self.resume_from)
        if self.load_from:
            runner.load_checkpoint(self.load_from)
        runner.run(self.data_loaders, self.runtime_cfg.runtime_cfg.workflow, self.runtime_cfg.runtime_cfg.total_epochs)
        
