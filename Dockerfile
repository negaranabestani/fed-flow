FROM python:3.10
ADD requirements.txt /fed-flow/
ADD requirements1.txt /fed-flow/
ADD requirements2.txt /fed-flow/
ADD heavy-requirements.txt /fed-flow/
WORKDIR /fed-flow/
RUN pip install -r requirements.txt
RUN pip install -r requirements1.txt
RUN pip install -r requirements2.txt
#RUN pip install -r heavy-requirements.txt
RUN pip install nvidia-cublas-cu11==11.10.3.66
RUN pip install nvidia-cuda-cupti-cu11==11.7.101
RUN pip install nvidia-cuda-nvrtc-cu11==11.7.99
RUN pip install nvidia-cuda-runtime-cu11==11.7.99
RUN pip install nvidia-cudnn-cu11==8.5.0.96
RUN pip install nvidia-cufft-cu11==10.9.0.58
RUN pip install nvidia-curand-cu11==10.2.10.91
RUN pip install nvidia-cusolver-cu11==11.4.0.1
RUN pip install nvidia-cusparse-cu11==11.7.4.91
RUN pip install nvidia-nccl-cu11==2.14.3
RUN pip install nvidia-nvtx-cu11==11.7.91
RUN pip install tensorboard==2.8.0
RUN pip install tensorboard-data-server==0.6.1
RUN pip install tensorboard-plugin-wit==1.8.1
RUN pip install tensordict==0.1.2
RUN pip install torch==2.0.1
RUN pip install torchrl==0.1.1
RUN pip install torchvision==0.15.2
RUN pip install tqdm==4.66.1
COPY app /fed-flow/app
COPY energy-estimation /fed-flow/energy
WORKDIR /fed-flow/