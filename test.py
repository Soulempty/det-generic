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

def get_path(txt_path,files=[]):
    data_path = os.path.dirname(txt_path)
    with open(txt_path) as f:
        for line in f.readlines():
            img_path = os.path.join(data_path,'source',line.strip())
            files.append(img_path)

def read_image(files):
    images = {}
    for f in files:
        img = cv2.imread(f)
        filename = os.path.basename(f)
        images[filename] = img
    return images 

def display(results,save_path):
    os.makedirs(save_path,exist_ok=True)
    for f in results:
        img = cv2.imread(f)
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

def test(data_path,model_path,gpu_id,save_path='save',score_thr=0.5):
    det = Detect(model_path)
    det.init_model(gpu_id=gpu_id)
    train_code = det.train_cls
    det.set_test_code(train_code)
    files = []
    if os.path.isfile(data_path):
        get_path(data_path,files)
    else:
        get_file(data_path,files)

    results = det.test(img_path=files,score_thr=score_thr)
    display(results,save_path)


if __name__ == "__main__":
    data_path = 'data/wxn/val.txt'  
    save_path = 'data/wxn/results'    
    model_path = "data/wxn/model/epoch_25.pth"
    gpu_id = 0
    score_thr = 0.5
    test(data_path,model_path,gpu_id,save_path,score_thr)
