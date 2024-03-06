FROM python:3.10
ADD requirements.txt /
ADD requirements1.txt /
ADD requirements2.txt /
ADD heavy-requirements.txt /
RUN pip --timeout=1000 install -r requirements.txt
RUN pip --timeout=1000 install -r requirements1.txt
#RUN pip --timeout=1000 install -r heavy-requirements.txt
RUN pip --timeout=10000 install nvidia-cublas-cu11==11.10.3.66
RUN pip --timeout=10000 install nvidia-cuda-cupti-cu11==11.7.101
RUN pip --timeout=10000 install nvidia-cuda-nvrtc-cu11==11.7.99
RUN pip --timeout=10000 install nvidia-cuda-runtime-cu11==11.7.99
RUN pip --timeout=10000 install nvidia-cudnn-cu11==8.5.0.96
RUN pip --timeout=10000 install nvidia-cufft-cu11==10.9.0.58
RUN pip --timeout=10000 install nvidia-curand-cu11==10.2.10.91
RUN pip --timeout=10000 install nvidia-cusolver-cu11==11.4.0.1
RUN pip --timeout=10000 install nvidia-cusparse-cu11==11.7.4.91
RUN pip --timeout=10000 install nvidia-nccl-cu11==2.14.3
RUN pip --timeout=10000 install nvidia-nvtx-cu11==11.7.91
RUN pip --timeout=10000 install tensorboard==2.8.0
RUN pip --timeout=10000 install tensorboard-data-server==0.6.1
RUN pip --timeout=10000 install tensorboard-plugin-wit==1.8.1
RUN pip --timeout=10000 install tensordict==0.1.2
RUN pip --timeout=10000 install torch==2.0.1
RUN pip --timeout=10000 install torchrl==0.1.1
RUN pip --timeout=10000 install torchvision==0.15.2
RUN pip --timeout=10000 install tqdm==4.66.1
RUN pip --timeout=10000 install -r requirements2.txt
COPY app /fed-flow/app
COPY energy-estimation /fed-flow/energy
WORKDIR /fed-flow/