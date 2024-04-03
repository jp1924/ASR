from nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /root
USER root

ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 서버 관련 유틸
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y ffmpeg wget net-tools build-essential git curl vim nmon && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip

# 파이썬 관련 유틸
RUN pip install -U pip wheel setuptools && \
    pip install transformers==4.39.1 accelerate==0.28.0 datasets==2.18.0 evaluate==0.4.1 && \
    pip install bitsandbytes==0.41.3.post2 scipy==1.12.0 sentencepiece==0.1.99 deepspeed==0.13.1 wandb==0.16.3 && \
    pip install soundfile librosa jiwer && \
    pip install setproctitle glances[gpu] && \
    pip install black flake8 isort && \
    pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install flash-attn==2.5.2 

RUN ln -s /root/workspace/.vscode /root/.vscode