#codeing:utf-8
import sys
import subprocess
import os
import pickle
import shutil
import os.path as osp
import cv2
import tempfile
import time
import numpy as np
import copy
import mmcv
import torch
import random
import logging
from xml.dom import minidom
from threading import Thread

import xml.etree.ElementTree as ET
import torch.distributed as dist
from mmcv.runner import get_dist_info
from contextlib import contextmanager

logger_initialized = {}
root_dir = os.path.join(os.path.dirname(__file__),"../..")
TestDataCfgPath = os.path.join(root_dir,"configs/_base_/datasets/test_data.json")
EvalDataCfgPath = os.path.join(root_dir,"configs/_base_/datasets/eval_data.json")
ModelCfgPath = os.path.join(root_dir,"configs/_base_/models/model.json")
TestCfgPath = os.path.join(root_dir,"configs/_base_/cfgs/test_cfg.json")


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield   
    if local_rank == 0:
        torch.distributed.barrier()


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger

def async_(f):
  def wrapper(*args, **kwargs):
    thr = Thread(target=f, args=args, kwargs=kwargs)
    thr.start()
 
  return wrapper

def replace_ImageToTensor(pipelines):
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines

def single_gpu_test(model,data_loader,show=False,out_dir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results



class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
 
def dict2obj(dict_item):
    if not isinstance(dict_item, dict):
        return dict_item
    inst=Dict()
    for k,v in dict_item.items():
        inst[k] = dict2obj(v)
    return inst


#???????????????nms?????????????????????         
def nms(boxes, threshold=0.05):
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
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)
def compute_iou(box, boxes, box_area, boxes_area):
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
    iou = intersection / (union+1e-6)
    return iou

def gen_voc(parse_info,save_path,file_name):
    
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
    with open(osp.join(save_path,os.path.splitext(file_name)[0]+".xml"), 'w') as x:
        x.write(doc.toprettyxml())
    x.close()


def data_split(data_path,lb_dir,img_dir,codes=[],r=0.8):

    train_path = os.path.join(data_path,"trainval.txt")
    val_path = os.path.join(data_path,"val.txt")
    test_path = os.path.join(data_path,"test.txt")

    train_txt = open(train_path,'w')
    test_txt = open(test_path,'w')
    val_txt = open(val_path,'w')

    tbs = {}
    tmp_img = []

    img_path = os.path.join(data_path,img_dir)
    files = os.listdir(img_path)

    t = os.path.join(data_path,'train.txt')
    if os.path.exists(t):
        files = [line.strip() for line in open(t).readlines()]
    
    for f in files:
        xml_path = os.path.join(data_path,lb_dir,f[:-4]+'.xml')
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objs = root.findall('object')
            code = ""
            flag = True
            anomaly = False
            if len(objs) > 0:
                for obj in objs:
                    name = obj.find('name').text
                    bnd_box = obj.find('bndbox')
                    if flag:
                        code = name
                        flag = False

                    box = [bnd_box.find('xmin').text,
                           bnd_box.find('ymin').text,
                           bnd_box.find('xmax').text,
                           bnd_box.find('ymax').text]
                    if "NaN" in box or float(box[0])<0 or float(box[1])<0 or float(box[2])<0 or float(box[3])<0:
                        anomaly = True
                if not anomaly:
                    if code not in tbs:
                        tbs[code] = [f]
                    else:
                        tbs[code].append(f)            
            else:
                tmp_img.append(f)
        else:
            tmp_img.append(f)
    
    for code in tbs:
        imgs = tbs[code]
        random.shuffle(imgs)
        if len(codes)>=1 and code not in codes:
            for x in imgs:
                test_txt.write(x+'\n')
            continue

        m = len(imgs)
        nn = int(m*r+0.5)
        i = 0
        for x in imgs:
            i += 1
            if i <=nn:
                train_txt.write(x+'\n')
            else:
                val_txt.write(x+'\n')
    
    for x in tmp_img:
        train_txt.write(x+"\n")
    train_txt.close()
    test_txt.close()
    val_txt.close()

    
def get_train_cls(data_path,lb_dir='label'):

    xml_path = os.path.join(data_path,lb_dir)
    files = os.listdir(xml_path)
    train_cls = []
    names = {}
    for f in files:
        xml_p = os.path.join(xml_path,f)
        tree = ET.parse(xml_p)
        root = tree.getroot()
        objs = root.findall('object')
        if len(objs) >0 :
            for obj in objs:
                name = obj.find('name').text
                names[name] = names.get(name,0)+1
    for k in names:
        if names[k] > 5:
            train_cls.append(k)
    print(train_cls)
    return train_cls
