ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data/detection
WORKDIR /data/detection
#COPY . /data/detection

RUN pip install pynvml terminaltables pycocotools pandas opencv-python addict numpy pyyaml yapf -i https://pypi.tuna.tsinghua.edu.cn/simple

#WORKDIR /data/detection/mmcv
#RUN pip install --no-cache-dir -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

#WORKDIR  /data/detection/mmdetection
#RUN conda clean --all
#ENV FORCE_CUDA="1"
#RUN pip install --no-cache-dir -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

#WORKDIR /data/detection


