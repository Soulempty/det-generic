import os

import mmcv
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class TestDataset(Dataset):
    """Custom dataset for detection.
    """
    def __init__(self,
                 code_cls,
                 pipeline,
                 data_root=None,
                 img_path=None,
                 cls_file='',
                 test_mode=True,
                 img_type="jpg"):   

        self.CLASSES = code_cls    
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        
        self.test_mode = test_mode   
        self.dict_flag = False
        if isinstance(img_path,str):
            if not os.path.exists(img_path):
                img_path = os.path.join(data_root,img_path)
            pipeline[0]["type"] = "LoadImageFromFile"

        elif isinstance(img_path,(list,tuple)):          
            if isinstance(img_path[0],str):
                pipeline[0]["type"] = "LoadImageFromFile"
            elif not isinstance(img_path[0],(dict,np.ndarray)):
                print("You should upload a list of cv::mat images or dict,not {} type".format(type(img_path[0]))) 
                img_path = os.path.join(self.data_root,"images")
            if isinstance(img_path[0],np.ndarray):
                img_path = [{"img":img,"filename":i } for i,img in enumerate(img_path)]
        elif isinstance(img_path,dict):
            self.dict_flag = True
        else:
            img_path = os.path.join(data_root,"images")

        self.data_infos = self.load_annotations(img_path)
        self.pipeline = Compose(pipeline)
    
    def load_annotations(self,img_path):
        data_infos = []
        
        if self.dict_flag:
            for filename in img_path:
                width,height = 0,0
                img = img_path[filename]
                try:
                    height,width = img.shape[:2]
                except Exception as e:
                    continue
                data_infos.append(dict(filename=filename, img=img, width=width, height=height))

        if isinstance(img_path,str) and os.path.isdir(img_path):
            fs = os.listdir(img_path)
            for f in fs:
                width,height = 0,0
                filename = os.path.join(img_path,f)
                try:
                    img = Image.open(filename)
                except Exception as e:
                    continue
                width, height = img.size 

                data_infos.append(dict(id=f.split('.')[0], cls_label='', filename=filename, width=width, height=height))
        if isinstance(img_path,(list,tuple)):
            if isinstance(img_path[0],str):
                for filename in img_path:
                    width,height = 0,0
                    basename = os.path.basename(filename)
                    try:
                        img = Image.open(filename)
                        #img = cv2.imread(filename)
                    except Exception as e:
                        print(e)
                        continue
                    width, height = img.size 

                    data_infos.append(dict(id=basename, cls_label='',filename=filename, width=width, height=height))
            elif isinstance(img_path[0],dict):
                for item in img_path:
                    width,height = 0,0
                    filename = item["filename"]
                    img = item["img"]
                    try:
                        height,width = img.shape[:2]
                    except Exception as e:
                        continue 
                    data_infos.append(dict(filename=filename, img=img, width=width, height=height))
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            img_info = self.data_infos[idx]
            results = dict(img_info=img_info)
            return self.pipeline(results)
