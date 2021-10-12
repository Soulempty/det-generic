#coding:utf-8
import argparse
import os
import time
import copy
import logging
from tqdm import tqdm
import json
import pynvml
import torch
import numpy as np
import gc


import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_state_dict

from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from .utils import replace_ImageToTensor, nms
from .utils import DataCfgPath, ModelCfgPath, TestCfgPath, NewCodePath, Newcodes
from .utils import DefaultGrade, BlurThreshold, CodeWithoutBlur, NFDScore

# infos about model and model set.

class TestInfo():
    def __init__(self,model_path=None): #
        self.model_path_ = model_path
        self.info_path = model_path.replace(os.path.splitext(model_path)[-1],'.json')
        self.model_infos = {}
        self.init_model_info()  #TODO add a json file to update model info.

    @property
    def model_path(self):
        return self.model_path_

    def add_code(self,add_codes={}):
        global Newcodes,NewCodePath
        step_id = self.model_info["step_id"]
        new_codes = Newcodes.get(step_id,{})
        if add_codes and len(add_codes)>0:
            new_codes.update(add_codes)
            Newcodes[step_id] = new_codes
            Newcodes.dump(NewCodePath)
        Newcodes = mmcv.Config.fromfile(NewCodePath)

    def get_default_grade(self):
        step_id = self.model_info["step_id"]
        train_codes = self.model_info['train_codes']
        default_grade = DefaultGrade.get(step_id,{})
        init_grade = {code:'G' for code in train_codes}
        if default_grade and len(default_grade)>0:
            mod_grade = {code:'GP' for code in train_codes if 'oth' not in code}
            init_grade.update(mod_grade)
            init_grade.update(default_grade)        
        return init_grade

    @property
    def model_info(self):
        return self.model_infos
    
    def init_model_info(self):       
        if os.path.exists(self.info_path):
            with open(self.info_path, 'r') as f:
                infos = json.load(f)
        else:
            infos = torch.load(self.model_path, map_location='cpu')["meta"]
            infos["model_state"] = "used"
            with open(self.info_path, 'w') as f:
                json.dump(infos,f,indent=4)
        train_codes = []
        for code in infos["CLASSES"]:
            txt = code.split('_')[0]
            if txt not in train_codes:
                train_codes.append(txt)
                oth_code = 'oth-' + txt
                if 'other' not in txt and oth_code not in train_codes:
                    train_codes.append(oth_code)
        
        step_id = infos.get("step_id",'12345')
        new_codes = Newcodes.get(step_id,{})
        if step_id[0] == '1':
            id_ = step_id[1]
            NFD = "A{}NFD".format(id_)
            NNC = "A{}NNC".format(id_)
            empty_size_code = new_codes.get('empty_size_code','AALMK')
            train_codes += [NFD,NNC,'A0NFP',empty_size_code] 
        elif step_id[0] == '4':
            id_ = step_id[1]
            NFD = "T{}NFD".format(id_)
            NNC = "T{}NNC".format(id_)
            empty_size_code = new_codes.get('empty_size_code','TAAAA')
            train_codes += [NFD,NNC,empty_size_code]
        elif step_id[0] == '2':
            NFD = new_codes.get('ok_code','CXNL1')
            empty_size_code = new_codes.get('empty_size_code','CXTG1')
            train_codes += [NFD,empty_size_code]
        else:
            NFD = new_codes.get('ok_code','CXNL1')  
            NNC = new_codes.get('blur_code','XXNNC')
            empty_size_code = new_codes.get('empty_size_code','CXTG1')
            train_codes += [NFD,NNC,empty_size_code]

        t = infos["time"]
        epoch = infos["epoch"]
        model_state = infos["model_state"]
        model_name = os.path.basename(self.model_path)
        self.model_infos = {"model_name":model_name,"train_codes":train_codes,"step_id":step_id,"epoch":epoch,"time":t,"model_state":model_state}


class Detect(): # 
    def __init__(self, model_path=None, samples_per_gpu=None, size_thr=800, grade_judge_code=None, gpu_id=0, blr_thr=8,blr_diff=4):
        super().__init__()
        self.model_path = model_path
        self.size_thr = size_thr
        self.grade_judge_code = grade_judge_code
        self.samples_per_gpu = samples_per_gpu  
        self.gpu_id = gpu_id
        self.blr_thr = blr_thr
        self.blr_diff = blr_diff
        self.score_thr_ = None
        self.default_grades = {}
        self.undead_codes = []
        self.code_prior = {}

        self.model = None
        self.model_infos = {}
        self.code_cls_ = []
        self.test_code_ = []
        self.list_flag = False

        self.data_cfg = mmcv.Config.fromfile(DataCfgPath)
        self.model_cfg = mmcv.Config.fromfile(ModelCfgPath)
        self.test_cfg = mmcv.Config.fromfile(TestCfgPath)
        self.init_cfg()
        self.model_init_flag = False
        self.first_score_flag = True

    def init_cfg(self):
       
        if self.samples_per_gpu:
            self.data_cfg["data"]["samples_per_gpu"] = self.samples_per_gpu
        if not self.model_path: 
            raise ValueError("You should give the path to the model!")
        elif not os.path.exists(self.model_path):
            raise ValueError("The model path you give does not exist")

    def modify_cfg(self, model_path=None, samples_per_gpu=None, size_thr=800, grade_judge_code=None, gpu_id=0):
        self.model_path = model_path
        self.size_thr = size_thr
        self.grade_judge_code = grade_judge_code
        self.samples_per_gpu = samples_per_gpu
        self.gpu_id = gpu_id
        self.init_cfg()

    @property
    def code_cls(self):
        return self.code_cls_

    @property
    def test_code(self):
        return self.test_code_
    @property
    def score_thr(self):
        return self.score_thr_

    @property
    def nfp_score_thr(self):
        return self.nfp_score_thr_

    @property
    def model_info(self):
        return self.model_infos

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


    def load_checkpoint(self):
        checkpoint = torch.load(self.model_path, map_location='cpu')
        infos = checkpoint["meta"]
        model_name = os.path.basename(self.model_path)
        step_id = infos.get("step_id",'11629')
        self.model_infos = {"model_name":model_name,
                            "model_state":'used',
                            "train_codes":infos["CLASSES"],
                            "step_id":step_id, 
                            "epoch":infos["epoch"],
                            "time":infos["time"]}
        self.blr_thr  = BlurThreshold.get(step_id,{}).get('bl_max',self.blr_thr)
        self.blr_diff  = BlurThreshold.get(step_id,{}).get('dif_max',self.blr_diff)
        self.default_grades = DefaultGrade.get(step_id,{})

        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {self.model_path}')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        return state_dict

    def init_model(self,gpu_id=None,use_fp16=True,use_benchmark=False):
        if gpu_id:
            self.gpu_id = gpu_id
        torch.cuda.set_device(self.gpu_id)
        if use_benchmark:
            torch.backends.cudnn.benchmark = True
        state_dict = self.load_checkpoint()
        self.code_cls_ = self.model_info["train_codes"]
        step_id = self.model_info['step_id']
        len_cls = len(self.code_cls)
       
        self.model_cfg['model']['roi_head']['bbox_head']['num_classes'] = len_cls
        self.model_cfg.model.pretrained = None

        self.model = build_detector(self.model_cfg.model, train_cfg=None, test_cfg=self.test_cfg.test_cfg)
        if use_fp16:
            print("Using fp16 to infer!\n")
            wrap_fp16_model(self.model)

        load_state_dict(self.model, state_dict)
        self.model.CLASSES = self.code_cls
        code_cls = []
        for code in self.code_cls:
            txt = code.split('_')[0]
            if txt not in code_cls:
                code_cls.append(txt)
                oth_code = 'oth-' + txt
                if 'other' not in txt and oth_code not in code_cls:
                    code_cls.append(oth_code)
        new_codes = Newcodes.get(step_id,{})
        if step_id[0] == '1':
            id_ = step_id[1]
            NFD = "A{}NFD".format(id_)
            NNC = "A{}NNC".format(id_)
            empty_size_code = new_codes.get('empty_size_code','AALMK')
            code_cls += [NFD,NNC,'A0NFP',empty_size_code]          

        elif step_id[0] == '4':
            id_ = step_id[1]
            NFD = "T{}NFD".format(id_)
            NNC = "T{}NNC".format(id_)
            empty_size_code = new_codes.get('empty_size_code','TAAAA')
            code_cls += [NFD,NNC,empty_size_code]
        elif step_id[0] == '2':
            NFD = new_codes.get('ok_code','CXNL1')
            empty_size_code = new_codes.get('empty_size_code','CXTG1')
            code_cls += [NFD,empty_size_code]
        else:
            NFD = new_codes.get('ok_code','CXNL1')  
            NNC = new_codes.get('blur_code','XXNNC')
            empty_size_code = new_codes.get('empty_size_code','CXTG1')
            code_cls += [NFD,NNC,empty_size_code]

        self.test_code_ = code_cls
        self.model = MMDataParallel(self.model, device_ids=[self.gpu_id])    
        self.model_init_flag = True
    
    def init_dataset(self,img_path):
        samples_per_gpu = self.data_cfg['data']['samples_per_gpu']
        if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            self.data_cfg['data']['test']['pipeline'] = replace_ImageToTensor(self.data_cfg['data']['test']['pipeline'])
        
        args = {'code_cls':self.code_cls}
        args.update(**self.data_cfg.data.test)
        args["img_path"] = img_path
        dataset = build_dataset(args)

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=self.data_cfg.data.workers_per_gpu,
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

    def test(self, img_path=None, score_thr=None, test_code=None, default_grade=None, undead_codes=None, code_prior=None, blr_thr=None,blr_diff=None):

        if img_path:
            if isinstance(img_path,list) and isinstance(img_path[0],np.ndarray):
                self.list_flag = True
        else:
            raise ValueError("You should give a valid img_path,it can be (1/ a path to all images 2/ a dict with style {filename:cvmat} 3/ a list of image path 4/ a list of dict('filename':'img.jpg','img':cvmat))")
        dataset, dataset_loader = self.init_dataset(img_path)
        
        if test_code:
            self.set_test_code(test_code)
        if not self.score_thr:
            self.set_code_score()
        if score_thr:
            self.set_code_score(score_thr)
        if blr_thr:
            self.blr_thr = blr_thr
        if blr_diff:
            self.blr_diff = blr_diff
        if default_grade:
            self.default_grades.update(default_grade)
        if undead_codes and len(undead_codes)>0:
            self.undead_codes = undead_codes
        if code_prior and len(code_prior)>0:
            self.code_prior = code_prior

        assert (self.model_init_flag == True), "before test,the model must be initialized(init_model)."
        self.model.eval()
        results = []
        prog_bar = mmcv.ProgressBar(len(dataset))
        infos = dataset.data_infos
        for i, data in enumerate(dataset_loader):
            blur_value = data["img_metas"][0].data[0][0]['blur_value']  
            blur_diff = data["img_metas"][0].data[0][0]['blur_diff']
            infos[i]['blur_value'] = blur_value
            infos[i]['blur_diff'] = blur_diff
            with torch.no_grad():
                result = self.model(return_loss=False, rescale=True,**data)
            results.extend(result)
            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
        results = self.std_results(results,dataset)
        del dataset
        del dataset_loader
        gc.collect()
        return results
    
    def std_results(self, results, dataset, ext_box=5,exist_score=0.002,other_blur_score=0.05):

        infos = dataset.data_infos
        code_cls = self.code_cls
        contents = {}
        step_id = self.model_info['step_id']
        id_ = step_id[1]
        NFD = "other"
        NNC = 'other'
        new_codes = Newcodes.get(step_id,{})
        if step_id[0] == '1':           
            NFD = "A{}NFD".format(id_)
            NNC = "A{}NNC".format(id_)
        elif step_id[0] == '4':
            NFD = "T{}NFD".format(id_)
            NNC = "T{}NNC".format(id_)
        elif step_id[0] == '2':
            NFD = new_codes.get('ok_code','CXNL1')  
            NNC = NFD     
        else:
            NFD = new_codes.get('ok_code','CXNL1')  
            NNC = new_codes.get('blur_code','XXNNC') 

        if self.score_thr.get(NFD,exist_score) < 0.01:
            exist_score = max(min(self.score_thr.get(NFD,exist_score),0.008),NFDScore.get(step_id,exist_score))
        if step_id[0] == '4' and self.score_thr.get(NNC,other_blur_score) < 0.06:
            other_blur_score = min(self.score_thr.get(NNC,other_blur_score),other_blur_score)

        if self.list_flag:
            contents = []

        for i, (res,info) in enumerate(zip(results,infos)):
            filename = info['filename']
            blur_value = info['blur_value']
            blur_diff = info['blur_diff']  
            blur = blur_value<self.blr_thr and blur_diff<self.blr_diff #TODO blur_diff
            if step_id[0] == '1':
                bl_param = BlurThreshold.get(step_id,{})
                bl_min = bl_param.get('bl_min',0.5)
                dif_min = bl_param.get('dif_min',0.1)
                blur = blur and blur_value>bl_min and blur_diff>dif_min 

            h,w = info['height'],info['width']
            objs = {'objects':[],'shape':(w,h)}
            obj = {} 
            tmp_obj = {}

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
            res_tmp_size = 0
            other_flag = False

            code_prior = 100  
            other_prior = 100
            grade_ = 'G'
            if N_ == 0:   
                tmp_obj['name'] = NFD      
            else:  
                inds = np.argsort(-scores)
                bboxes = bboxes[inds, :]
                labels = labels[inds]     

                
                for bbox, label in zip(bboxes, labels):
                    conf = bbox[-1]

                    bbox_int = bbox.astype(np.int32)
                    xmin, ymin = (bbox_int[0], bbox_int[1])
                    xmax, ymax = (bbox_int[2], bbox_int[3])
                    siz = np.sqrt((xmax-xmin)*(ymax-ymin))
                    txt = code_cls[label]

                    code = txt.split('_')[0]
                    grade = txt.split('_')[1] if len(txt.split('_'))>1 else 'G'

                    score_thr = self.score_thr.get(code,0.95)
                    
                    if code in self.test_code and conf >= score_thr:
                        
                        prior = self.code_prior.get(code,100)
                        if prior < code_prior:
                            code_prior = prior
                            res_code_size = max(siz,1)
                            obj["bndbox"] = [max(xmin-ext_box,0),max(ymin-ext_box,0),min(xmax+ext_box,w),min(ymax+ext_box,h)]
                            obj["name"] = code
                            obj["conf"] = conf
                            grade_ = grade
                        else:
                            if prior == code_prior and siz>res_code_size:                           
                                res_code_size = max(siz,1)
                                obj["bndbox"] = [max(xmin-ext_box,0),max(ymin-ext_box,0),min(xmax+ext_box,w),min(ymax+ext_box,h)]
                                obj["name"] = code
                                obj["conf"] = conf
                                grade_ = grade
                    else:#TODO BY SCORE.
                        other_flag = True   
                        prior = self.code_prior.get(code,100)
                        if prior < other_prior:
                            other_prior = prior 
                            res_tmp_size = max(siz,1)    
                            tmp_obj["bndbox"] = [max(xmin-ext_box,0),max(ymin-ext_box,0),min(xmax+ext_box,w),min(ymax+ext_box,h)]
                            tmp_obj["name"] = code #'other' 
                            tmp_obj["conf"] = conf  
                        else:
                            if prior == other_prior and siz>res_tmp_size: 
                                res_tmp_size = max(siz,1)
                                tmp_obj["bndbox"] = [max(xmin-ext_box,0),max(ymin-ext_box,0),min(xmax+ext_box,w),min(ymax+ext_box,h)]
                                tmp_obj["name"] = code #'other' 
                                tmp_obj["conf"] = conf  
                        
                                   
            objs['obj_size'] = round(res_tmp_size/6,3)
            if res_code_size>0 and len(obj)>0:
                obj['nfp_flag'] = False  
                objs["objects"].append(obj)
                objs["grade"] = grade_
                objs['obj_size'] = round(res_code_size/6,3)
            elif other_flag and len(tmp_obj)>0: #1 not test code  2 low score code 
                code_tmp = tmp_obj["name"]
                conf_tmp = tmp_obj.get('conf',0.5)     
                
                if step_id[0] == '1':
                    if "other" in code_tmp:
                        tmp_obj["name"] = "A0NFP"
                        tmp_obj['nfp_flag'] = False            
                    else:       
                        code = 'oth-' + code_tmp
                        score_thr = self.score_thr.get(code,0.95)
                        if code in self.test_code:
                            if score_thr == 1.0:
                                tmp_obj["name"] = "A0NFP"
                            else:
                                tmp_obj["name"] = code
                            tmp_obj['nfp_flag'] = False
                        else:  
                            if code_tmp in self.undead_codes:
                                tmp_obj['nfp_flag'] = True
                            else:
                                tmp_obj['nfp_flag'] = False
                            tmp_obj["name"] = 'other'

                    if blur and conf_tmp<other_blur_score and code[-3:] not in CodeWithoutBlur:
                        tmp_obj = {}
                        tmp_obj['nfp_flag'] = False
                        tmp_obj["name"] = NNC
                    
                else:                      
                    tmp_obj["name"] = 'other'
                    tmp_obj['nfp_flag'] = False
                    code = 'oth-' + code_tmp
                    if code in self.test_code:    
                        tmp_obj["name"] = code

                    if blur and conf_tmp<other_blur_score and res_tmp_size<50:
                        tmp_obj = {}               
                        tmp_obj['nfp_flag'] = False                           
                        tmp_obj["name"] = NNC

                objs["objects"].append(tmp_obj)
                objs["grade"] = "G"
            elif len(tmp_obj)>0:
                tmp_obj['nfp_flag'] = False               
                if blur:                       
                    tmp_obj["name"] = NNC
                objs["objects"].append(tmp_obj)
                objs["grade"] = "G"
            else:
                tmp_obj['name'] = 'other'
                tmp_obj['nfp_flag'] = False
                objs["objects"].append(tmp_obj)
                objs["grade"] = "G"

            if objs["objects"][0]['name'] == 'other' and "bndbox" in objs["objects"][0] and (not objs["objects"][0]['nfp_flag']):
                del objs["objects"][0]["bndbox"] 

            #TODO
            # if res_code_size>self.size_thr and objs["grade"]!='S' and objs["grade"]!='P':  
            #     objs["grade"] = "N"
            
            ccode = objs["objects"][0]['name']
            grade_df = self.default_grades.get(ccode,objs["grade"])
            if grade_df != "GP" or len(grade_df)!=2:
                objs["grade"] = grade_df
            if self.list_flag:
                
                contents.append(objs)
            else:
                contents[filename] = objs
        return contents
