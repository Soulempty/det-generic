#coding:utf-8

import os
import sys
import cv2
import numpy as np
import random
from time import time 
from glob import glob
from utils import gen_voc
import pandas as pd
from mmdet.apis import Detect


def get_file(path,files=[],ext=['.jpg','.JPG']):
    fs = os.listdir(path)
    for f in fs:
        p = os.path.join(path,f)
        if os.path.isdir(p):
            get_file(p,files)
        else:
            if os.path.splitext(p)[1] in ext:
                files.append(p)

def read_image(files):
    images = {}
    for f in files:
        img = cv2.imread(f)
        filename = os.path.basename(f)
        images[filename] = img
    return images 

def display(files,results,save_path):
    if not os.path.exists(save_path):
         os.makedirs(save_path)
    flag = False
    if files == None:
        files = results
        flag = True
    for f in files:
        img = None
        if flag:
            img = cv2.imread(f)
        else:
            img = files[f]
        basename = os.path.basename(f)
        res = results[f]
        for obj in res['objects']:   
            code = obj['name'] 
            conf = obj.get('conf',0.99)
            
            if 'bndbox' in obj:         
                bndbox = obj['bndbox']
                xmin,ymin = bndbox[:2]
                xmax,ymax = bndbox[2:]
               
                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255),2)
                name = code + f'|{conf:.3f}'
                cv2.putText(img, name, (max(xmin,0), max(ymin -2,0)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),1)  
        cv2.imwrite(os.path.join(save_path,basename),img)

def test(data_path,model_path,gpu_id,save_path='save',score_thr=0.2,test_style='image'):
    det = Detect(model_path)
    det.init_model(gpu_id=gpu_id)
    train_code = det.train_cls
    det.set_test_code(train_code)
    files = []
    get_file(data_path,files)

    if test_style == "file":
        results = det.test(img_path=files,score_thr=score_thr)
        display(None,results,save_path)
    if test_style == 'image':
        images = read_image(files)
        results = det.test(img_path=images,score_thr=score_thr) 
        display(images,results,save_path)

if __name__ == "__main__":
    data_path = 'data/test/bdd100'  
    save_path = 'data/res/bdd100'    
    model_path = "data/train/bdd100/model/epoch_25.pth"
    gpu_id = 1
    score_thr = 0.2
    test_code = None
    test(data_path,model_path,gpu_id,save_path,score_thr)
