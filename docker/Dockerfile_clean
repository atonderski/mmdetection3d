FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Lock a specific opencv version (bugfix)
RUN pip install -U opencv-python==4.5.5.64

# Install our extra dependencies
RUN pip install openmim zod open3d wandb

# Install MMDetection3D requirements
# RUN conda clean --all
COPY . /mmdetection3d
WORKDIR /mmdetection3d
RUN mim install --no-cache-dir -e .
ENV PYTHONPATH='.'
