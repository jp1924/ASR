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

# transformers는 #35196 PR기준
RUN pip install -U pip wheel setuptools && \
    pip install git+https://github.com/huggingface/transformers.git@d5aebc64653d09660818109f2fac55b5e1031023 && \
    pip install accelerate datasets evaluate trl peft deepspeed && \
    pip install kss bitsandbytes scipy sentencepiece librosa jiwer soundfile torch-audiomentations && \
    pip install ruff natsort setproctitle glances[gpu] wandb comet-ml cmake

RUN pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install flash-attn==2.7.0.post2
