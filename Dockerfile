# ==========================================
# Base: CUDA 11.8 + Ubuntu 20.04
# ==========================================
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV TCNN_CUDA_ARCHITECTURES=75

# ==========================================
# System Packages + Python 3.7 + nano
# ==========================================
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        git wget curl unzip ffmpeg \
        build-essential cmake ninja-build \
        libgl1 libglib2.0-0 libopencv-dev \
        python3.7 python3.7-dev python3.7-distutils \
        python3-pip nano \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드 (Python 3.7 기준)
RUN python3.7 -m pip install --upgrade pip==22.3.1 setuptools wheel

# ==========================================
# PyTorch (CUDA 11.6/11.8 호환)
# ==========================================
RUN python3.7 -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# ==========================================
# Python Requirements
# ==========================================
RUN python3.7 -m pip install tqdm scikit-learn matplotlib opencv-python plyfile==0.8.1

# ==========================================
# Third-party PyTorch Extensions
# ==========================================
WORKDIR /workspace

# diff-gaussian-rasterization
COPY third_party/diff-gaussian-rasterization ./third_party/diff-gaussian-rasterization
RUN TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST python3.7 -m pip install ./third_party/diff-gaussian-rasterization

# simple-knn
COPY third_party/simple-knn ./third_party/simple-knn
RUN TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST python3.7 -m pip install ./third_party/simple-knn

# ==========================================
# GLM 라이브러리 설치
# ==========================================
RUN apt-get update && apt-get install -y libglm-dev

# simple-diff-gaussian-rasterization
COPY third_party/simple-diff-gaussian-rasterization ./third_party/simple-diff-gaussian-rasterization
RUN TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST python3.7 -m pip install ./third_party/simple-diff-gaussian-rasterization

# ==========================================
# 추가 Python 패키지
# ==========================================
RUN python3.7 -m pip install \
    jaxtyping==0.2.11 \
    open_clip_torch==2.20.0 \
    timm==0.6.13 \
    ftfy==6.1.1 \
    regex==2023.10.3 \
    safetensors==0.3.1

# ==========================================
# Git 패키지 설치
# ==========================================
RUN python3.7 -m pip install "git+https://github.com/openai/CLIP.git"
RUN python3.7 -m pip install "git+https://github.com/facebookresearch/segment-anything.git"
RUN python3.7 -m pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# ==========================================
# COLMAP 설치
# ==========================================
RUN apt-get update && apt-get install -y colmap && rm -rf /var/lib/apt/lists/*

# ==========================================
# 환경 변수 + PYTHONPATH 설정
# ==========================================
ENV PYTHONPATH=/workspace:/workspace/third_party

# ==========================================
# 컨테이너 기본 설정
# ==========================================
CMD ["/bin/bash"]
