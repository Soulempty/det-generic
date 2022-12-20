## Introduction

本代码库是通过对MMDetection开源检测框架进行接口化、模块化设计，实现基于接口调用形式的训练与测试，非常方便于在工业应用中部署与落地，此代码库是通用性检测代码，相比ADC的代码，更加通用。

## 环境构建

-  下载当前代码库到服务器指定路径
-  使用docker构建镜像，命令如下：`docker build -t detect:v1 .`
-  基于构建镜像，运行docker容器，命令如下：`sudo docker run --runtime=nvidia -it --ipc=host -v /data/jiaochao/adc/det-generic:/data/detection/ --name det-generic detect:v1 /bin/bash`
-  编译mmcv: `cd mmcv && pip install -e .`
-  编译mmdet: `cd mmdetection && pip install -e .`
-  docker容器环境构建完成。

## 模型训练

-  将数据放到工作路径下，如`cp -r /data/jiaochao/data/wxn /data/jiaochao/adc/det-generic/data`
-  修改train.py文件末尾处的`data_dir="/data/detection/data/wxn"`,修改训练尺寸，如`img_size = [1560,1280]`
-  在容器中，执行如下命令：`python train.py`
-  在容器中，进行多卡训练：`CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 train.py  --distributed`

## 模型测试

-  在test.py文件末尾，修改测试数据路径（或txt文件），模型加载路径以及测试结果保存路径。
-  执行测试命令，`python test.py`

##  核心代码

-  mmdetection/mmdet/apis/train.py
-  mmdetection/mmdet/apis/test.py
-  mmcv/mmcv/runner/epoch_based_runner.py
