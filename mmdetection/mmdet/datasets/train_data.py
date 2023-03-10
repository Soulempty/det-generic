#coding:utf-8
import os
import mmcv
import numpy as np
from PIL import Image
import pandas as pd
import xml.etree.ElementTree as ET
from mmdet.core import eval_map, eval_recalls, confusion_matrix

from .custom import CustomDataset
from .builder import DATASETS
from terminaltables import AsciiTable

import csv
import cv2
from xml.dom import minidom
from collections import Counter

@DATASETS.register_module()
class TrainDataset(CustomDataset):

    def __init__(self, code_cls, min_size=0, **kwargs):
        super(TrainDataset, self).__init__(**kwargs)
        self.CLASSES = code_cls
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        data_infos = []
        cls_flag = False
        if os.path.isfile(self.cls_file):
            df = pd.read_csv(self.cls_file)
            name_cls = df.set_index(['filename'])['code'].to_dict()
            cls_flag = True
        if self.img_path and os.path.isdir(self.img_path):
            self.img_prefix = ""
            fs = os.listdir(self.img_path)
            for f in fs:
                width,height = 0,0
                filename = os.path.join(self.img_path,f)
                try:
                    img = Image.open(filename)
                except Exception as e:
                    print(e,"error.")
                    continue
                width, height = img.size 
                if cls_flag:
                    cls_ = name_cls[f]
                else:
                    cls_ = ''
                data_infos.append(dict(id=f.split('.')[0], cls_label=cls_, filename=filename, width=width, height=height))
        elif isinstance(self.img_path,(list,tuple)):
            for filename in self.img_path:
                width,height = 0,0
                basename = os.path.basename(filename)
                try:
                    img = Image.open(filename)
                except Exception as e:
                    continue
                width, height = img.size 
                if cls_flag:
                    cls_ = name_cls[f]
                else:
                    cls_ = ''
                data_infos.append(dict(id=basename.split('.')[0], cls_label=cls_, filename=filename, width=width, height=height))
        else:
            items = mmcv.list_from_file(ann_file)
            for item in items:
                width,height = 0,0
                filename = '{}/{}'.format(self.img_dir,item)
                img_path = os.path.join(self.img_prefix, filename)
                img_id = item[:-4]
                try:
                    img = Image.open(img_path)
                except Exception as e:
                    print(e,"error.")
                    continue
                width, height = img.size  
                if cls_flag:
                    cls_ = name_cls[item]
                else:
                    cls_ = ''
                data_infos.append(dict(id=img_id, cls_label=cls_, filename=filename, width=width, height=height))
        return data_infos
        
    def get_subset_by_classes(self):
        """Filter imgs by user-defined categories
        """
        subset_data_infos = []
        for data_info in self.data_infos:
            img_id = data_info['id']
            xml_path = os.path.join(self.img_prefix, 'label',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in self.CLASSES:
                    subset_data_infos.append(data_info)
                    break

        return subset_data_infos
        
    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        xml_path = os.path.abspath(os.path.join(self.img_prefix, self.label_dir,'{}.xml'.format(img_id)))
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []

        if not os.path.exists(xml_path):
            
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name').text

                if name not in self.CLASSES:
                    continue
                label = self.cat2label[name]
                difficult = 0 #int(obj.find('difficult').text)
                bnd_box = obj.find('bndbox')
                bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
                ]
    
                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
                if difficult or ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
            if not bboxes:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0, ))
            else:
                bboxes = np.array(bboxes, ndmin=2) - 1 #2 dim array  n,4
                labels = np.array(labels)              #1 dim array  n,
            if not bboxes_ignore:
                bboxes_ignore = np.zeros((0, 4))
                labels_ignore = np.zeros((0, ))
            else:
                bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
                labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def gen_voc(self,parse_info,save_path,file_name):
    
        w,h = parse_info['size']
        objects = parse_info['objects']

        doc = minidom.Document()

        annotation = doc.createElement("annotation")
        doc.appendChild(annotation)
        folder = doc.createElement('folder')
        folder.appendChild(doc.createTextNode("wxn-v3"))
        annotation.appendChild(folder)

        filename = doc.createElement('filename')
        filename.appendChild(doc.createTextNode(file_name))
        annotation.appendChild(filename)

        source = doc.createElement('source')
        database = doc.createElement('database')
        database.appendChild(doc.createTextNode("Unknown"))
        source.appendChild(database)

        annotation.appendChild(source)

        size = doc.createElement('size')
        width = doc.createElement('width')
        width.appendChild(doc.createTextNode(str(w)))
        size.appendChild(width)
        height = doc.createElement('height')
        height.appendChild(doc.createTextNode(str(h)))
        size.appendChild(height)
        depth = doc.createElement('depth')
        depth.appendChild(doc.createTextNode(str(3)))
        size.appendChild(depth)
        annotation.appendChild(size)

        segmented = doc.createElement('segmented')
        segmented.appendChild(doc.createTextNode("0"))
        annotation.appendChild(segmented)

        for obj in objects:
        
            name = obj['name']
            x_min, y_min, x_max, y_max = obj['box']
            conf = obj["conf"]
            object = doc.createElement('object')
            nm = doc.createElement('name')
            nm.appendChild(doc.createTextNode(name))
            object.appendChild(nm)
            pose = doc.createElement('pose')
            pose.appendChild(doc.createTextNode("Unspecified"))
            object.appendChild(pose)
            truncated = doc.createElement('truncated')
            truncated.appendChild(doc.createTextNode("1"))
            object.appendChild(truncated)
            difficult = doc.createElement('difficult')
            difficult.appendChild(doc.createTextNode("0"))
            object.appendChild(difficult)
            bndbox = doc.createElement('bndbox')
            xmin = doc.createElement('xmin')
            xmin.appendChild(doc.createTextNode(str(x_min)))
            bndbox.appendChild(xmin)
            ymin = doc.createElement('ymin')
            ymin.appendChild(doc.createTextNode(str(y_min)))
            bndbox.appendChild(ymin)
            xmax = doc.createElement('xmax')
            xmax.appendChild(doc.createTextNode(str(x_max)))
            bndbox.appendChild(xmax)
            ymax = doc.createElement('ymax')
            ymax.appendChild(doc.createTextNode(str(y_max)))
            bndbox.appendChild(ymax)
            score = doc.createElement('score')
            score.appendChild(doc.createTextNode(str(conf)))
            bndbox.appendChild(score)
            object.appendChild(bndbox)
            annotation.appendChild(object)
        with open(os.path.join(save_path,file_name.replace(os.path.splitext(file_name)[-1],'.xml')), 'w') as x:
            x.write(doc.toprettyxml())
        x.close()   
                         
    def nms(self,boxes, threshold=0.2):
        """Performs non-maximum supression and returns indicies of kept boxes.
        boxes: [N, (x1, y1, x2, y2, score)]. Notice that (y2, x2) lays outside the box.
        scores: 1-D array of box scores.
        threshold: Float. IoU threshold to use for filtering.
        """
        
        if boxes.dtype.kind != "f":
            boxes = boxes.astype(np.float32)
    
        # Compute box areas
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, -1]
        area = (y2 - y1) * (x2 - x1)
    
        # Get indicies of boxes sorted by scores (highest first)
        ixs = scores.argsort()[::-1]
    
        pick = []
        while len(ixs) > 0:
            # Pick top box and add its index to the list
            i = ixs[0]
            pick.append(i)
            # Compute IoU of the picked box with the rest
            iou = self.compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
            # Identify boxes with IoU over the threshold. This
            # returns indicies into ixs[1:], so add 1 to get
            # indicies into ixs.
            remove_ixs = np.where(iou > threshold)[0] + 1
            # Remove indicies of the picked and overlapped boxes.
            ixs = np.delete(ixs, remove_ixs)
            ixs = np.delete(ixs, 0)
        return np.array(pick, dtype=np.int32)
    def compute_iou(self,box, boxes, box_area, boxes_area):
        """Calculates IoU of the given box with the array of the given boxes.
        box: 1D vector [y1, x1, y2, x2]
        boxes: [boxes_count, (y1, x1, y2, x2)]
        box_area: float. the area of 'box'
        boxes_area: array of length boxes_count.
    
        Note: the areas are passed in rather than calculated here for
              efficency. Calculate once in the caller to avoid duplicate work.
        """
        # Calculate intersection areas
        y1 = np.maximum(box[0], boxes[:, 0])
        y2 = np.minimum(box[2], boxes[:, 2])
        x1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box_area + boxes_area[:] - intersection[:]
        iou = intersection / union
        return iou
