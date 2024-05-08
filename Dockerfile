from nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /root
USER root

ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 서버 관련 유틸
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y ffmpeg wget net-tools build-essential git curl vim nmon tmux && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip

# 파이썬 관련 유틸
RUN pip install -U pip wheel setuptools && \
    pip install transformers==4.39.3 accelerate==0.29.2 datasets==2.18.0 evaluate==0.4.1 && \
    pip install trl==0.8.1 peft==0.10.0 bitsandbytes==0.41.3.post2 scipy==1.12.0 sentencepiece==0.2.0 deepspeed==0.14.0 wandb==0.16.6 && \
    pip install soundfile librosa jiwer && \
    pip install setproctitle glances[gpu] && \
    pip install ruff natsort && \
    pip install torch==2.2.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install flash-attn==2.5.7