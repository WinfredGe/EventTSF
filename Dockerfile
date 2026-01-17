FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt update && \
    apt install -y \
        wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
        libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /temp
RUN wget https://www.python.org/ftp/python/3.11.10/Python-3.11.10.tgz && \
    tar -xvf Python-3.11.10.tgz
RUN cd Python-3.11.10 && \
    ./configure --enable-optimizations && \
    make && \
    make install
WORKDIR /workspace
RUN rm -r /temp && \
    ln -s /usr/local/bin/python3 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3 /usr/local/bin/pip
RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121 && \
    rm -r /root/.cache/pip
RUN python -m pip install --upgrade pip
RUN pip install hydra-core
RUN pip install pytorch_lightning==2.5.0
RUN pip install lightning
RUN pip install wandb
RUN pip install jupyterlab
EXPOSE 8888
CMD ["bash", "-c", "sleep infinity"]
