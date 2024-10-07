FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /root
USER root

ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 서버 관련 유틸
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y ffmpeg wget net-tools build-essential git curl vim nmon tmux && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip

RUN ln -s /usr/bin/python3.10 /usr/bin/python

RUN pip install -U pip wheel setuptools && \
    pip install --no-cache-dir git+https://github.com/huggingface/transformers.git@78b2929c0554b79e0489b451ce4ece14d265ead2 && \
    pip install accelerate==0.33.0 datasets==2.21.0 evaluate==0.4.2 peft==0.12.0 deepspeed==0.15.0 fire==0.7.0 && \
    pip install kss==6.0.4 bitsandbytes==0.43.3 scipy==1.14.1 sentencepiece==0.2.0 librosa==0.10.2 jiwer==3.0.4 soundfile==0.12.1 torch-audiomentations==0.11.1 && \
    pip install ruff natsort setproctitle glances[gpu] wandb comet-ml cmake

RUN pip install torch==2.4.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install flash-attn==2.6.3