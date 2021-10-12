#coding:utf-8
from .test import TestInfo, Detect
from .train import TrainInfo, Detection
from .utils import nms, replace_ImageToTensor, single_gpu_test, multi_gpu_test

__all__ = ['TestInfo', 'Detect', 'TrainInfo', 'Detection','nms', 'replace_ImageToTensor', 'single_gpu_test','multi_gpu_test']
