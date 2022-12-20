#coding:utf-8
import os
import time
import copy
import json
import argparse
import logging
from tqdm import tqdm

import torch
import gc
import pynvml
import numpy as np

import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_state_dict
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from .utils import replace_ImageToTensor, nms
from .utils import TestDataCfgPath, ModelCfgPath, TestCfgPath


class Detect(): # 
    def __init__(self, model_path=None, samples_per_gpu=None, blr_det=False, blr_vmean_thr=1.0,blr_vvar_thr=0.116):
        super().__init__()
        self.model_path = model_path
        self.samples_per_gpu = samples_per_gpu  
        self.blr_det = blr_det
        self.blr_vmean_thr = blr_vmean_thr
        self.blr_vvar_thr = blr_vvar_thr
        self.score_thr_ = None
        self.code_prior = {}
        self.train_cls = []
        self.test_code_ = []
        self.model = None
        self.list_flag = False
        self.model_init_flag = False

        self.data_cfg = mmcv.Config.fromfile(TestDataCfgPath)
        self.model_cfg = mmcv.Config.fromfile(ModelCfgPath)
        self.test_cfg = mmcv.Config.fromfile(TestCfgPath)
        self.init_cfg()
        

    def init_cfg(self):
       
        if self.samples_per_gpu:
            self.data_cfg["data"]["samples_per_gpu"] = self.samples_per_gpu
        if not self.model_path: 
            raise ValueError("You should give the path to the model!")
        elif not os.path.exists(self.model_path):
            raise ValueError("The model path you give does not exist!")
        self.data_cfg["data"]["test"]["pipeline"][1]["transforms"][0]["flag"] = self.blr_det
        self.data_cfg.dump(TestDataCfgPath)

    @property
    def test_code(self):
        return self.test_code_

    @property
    def score_thr(self):
        return self.score_thr_

    def release(self):
        if self.model_init_flag:
            self.model = self.model.cpu()
            del self.model
            torch.cuda.empty_cache() 
            time.sleep(2)
            return
        else:
            return 
    
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

    def init_model(self, gpu_id=None, use_fp16=True, use_benchmark=False):
        torch.cuda.set_device(gpu_id)
        if use_benchmark:
            torch.backends.cudnn.benchmark = True
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        infos = checkpoint["meta"]
        self.train_cls = infos["CLASSES"]
        len_cls = len(self.train_cls)

        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {self.model_path}')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

        self.model_cfg['model']['roi_head']['bbox_head']['num_classes'] = len_cls
        self.model_cfg.model.pretrained = None

        self.model = build_detector(self.model_cfg.model, train_cfg=None, test_cfg=self.test_cfg.test_cfg)
        if use_fp16:
            print("Using fp16 to infer!\n")
            wrap_fp16_model(self.model)

        load_state_dict(self.model, state_dict)
        self.model = MMDataParallel(self.model, device_ids=[gpu_id])    
        self.model_init_flag = True
    

    def init_dataset(self,img_path):
        samples_per_gpu = self.data_cfg['data']['samples_per_gpu']
        if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            self.data_cfg['data']['test']['pipeline'] = replace_ImageToTensor(self.data_cfg['data']['test']['pipeline'])
        
        args = {'code_cls':self.train_cls}
        args.update(**self.data_cfg.data.test)
        args["img_path"] = img_path
        dataset = build_dataset(args)
        workers = self.data_cfg.data.workers_per_gpu
        if self.blr_det:
            workers = 0
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=workers,
            dist=False,
            shuffle=False)
        return dataset, data_loader

    def set_code_score(self,score_thr=0.95):
        if self.score_thr == None:
            self.score_thr_ = {}
        if isinstance(score_thr,dict):
            self.score_thr_.update(score_thr)
        elif isinstance(score_thr,float):
            self.score_thr_.update({code:score_thr for code in self.test_code})             
        else:
            print(self.score_thr)

    def set_test_code(self,test_code):
        if test_code:
            self.test_code_ = test_code

    def test(self, img_path=None, score_thr=None, test_code=None, code_prior=None, blr_vmean_thr=None,blr_vvar_thr=None):

        if img_path:
            if isinstance(img_path,list) and isinstance(img_path[0],np.ndarray):
                self.list_flag = True
        else:
            raise ValueError("You should give a valid img_path,it can be (1/ a path to all images 2/ a dict with style {filename:cvmat} 3/ a list of image path 4/ a list of dict('filename':'img.jpg','img':cvmat))")
        dataset, dataset_loader = self.init_dataset(img_path)
        
        if test_code:
            self.set_test_code(test_code)
        if score_thr:
            self.set_code_score(score_thr)
        else:
            self.set_code_score()
        if blr_vmean_thr:
            self.blr_vmean = blr_vmean_thr
        if blr_vvar_thr:
            self.blr_vvar_thr = blr_vvar_thr

        if code_prior and len(code_prior)>0:
            self.code_prior.update(code_prior)
        assert (self.model_init_flag == True), "before test,the model must be initialized(init_model)."
        
        self.model.eval()
        results = []
        prog_bar = mmcv.ProgressBar(len(dataset))
        infos = dataset.data_infos
        for i, data in enumerate(dataset_loader):
            
            infos[i]['blr_vmean'] = data["img_metas"][0].data[0][0].get('blr_vmean',0) 
            infos[i]['blr_vvar'] = data["img_metas"][0].data[0][0].get('blr_vvar',0)
            with torch.no_grad():
                result = self.model(return_loss=False, rescale=True,**data)
            results.extend(result)
            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
        results = self.std_results(results,infos)
        del dataset
        del dataset_loader
        return results
    
    def std_results(self, results, infos, ext_box=2,exist_score=0.002,blr_score=0.2):
        contents = {}      
        NFD = "NFD"
        NNC = 'NNC'
        if self.list_flag:
            contents = []
        for _, (res,info) in enumerate(zip(results,infos)):
            filename = info['filename']
            blr_vmean = info['blr_vmean']
            blr_vvar = info['blr_vvar']  
            blur = blr_vmean<self.blr_vmean_thr and blr_vvar<self.blr_vvar_thr 
            w = info['width']
            h = info['height']
            objs = {'objects':[],'shape':(w,h)}
            bboxes = np.vstack(res)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(res)]
            labels = np.concatenate(labels)
            if len(bboxes)>0:
                assert bboxes.shape[1] == 5
                inds = nms(bboxes)
                bboxes = bboxes[inds, :]
                labels = labels[inds]
            scores = bboxes[:, -1]
            inds = scores>=exist_score
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            N_ = len(bboxes)
            res_code_size = 0
            code_prior = 100  
            if N_ == 0:   
                objs["objects"].append({"name":NFD})    
            else:  
                inds = np.argsort(-scores)
                bboxes = bboxes[inds, :]
                labels = labels[inds]                  
                for bbox, label in zip(bboxes, labels):
                    obj = {} 
                    tmp_obj = {}
                    conf = round(bbox[-1],3)
                    bbox_int = bbox.astype(np.int32)
                    xmin, ymin = (bbox_int[0], bbox_int[1])
                    xmax, ymax = (bbox_int[2], bbox_int[3])
                    siz = np.sqrt((xmax-xmin)*(ymax-ymin))
                    code = self.train_cls[label]
                    score_thr = self.score_thr.get(code,0.95)                 
                    if code in self.test_code and conf >= score_thr:              
                        prior = self.code_prior.get(code,100)
                        if prior < code_prior:
                            code_prior = prior
                            res_code_size = max(siz,1)                           
                        else:
                            if prior == code_prior and siz>res_code_size:                           
                                res_code_size = max(siz,1)
                        obj["bndbox"] = [max(xmin-ext_box,0),max(ymin-ext_box,0),min(xmax+ext_box,w),min(ymax+ext_box,h)]
                        obj["name"] = code
                        obj["conf"] = conf
                        objs["objects"].append(obj)

            if blur and self.blr_det and ("conf" not in objs["objects"] or  "conf" in objs["objects"] and objs["objects"][0]["conf"]<blr_score):
                    tmp_obj = {}
                    tmp_obj["name"] = NNC     
            if self.list_flag:           
                contents.append(objs)
            else:
                contents[filename] = objs
        return contents
