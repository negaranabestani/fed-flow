FROM python:3.10.6

RUN echo ${PWD}
RUN mkdir -p /fed-flow/app
#RUN mkdir -p /fed-flow/bin
#RUN mkdir -p /fed-flow/include
#RUN mkdir -p /fed-flow/lib
#RUN mkdir -p /fed-flow/lib64


ADD requirements.txt /fed-flow/

#RUN apt install python3.10-venv
#WORKDIR /fed-flow/
#RUN python3 -m pip install virtualenv
#COPY bin /fed-flow/bin
#COPY include /fed-flow/include
#COPY lib /fed-flow/lib
#COPY lib64 /fed-flow/lib64
#COPY pyvenv.cfg /fed-flow/
##WORKDIR /fed-flow/bin/
#RUN source bin/activate
WORKDIR /fed-flow/
#RUN pip install nvidia-cusparse-cu11
#RUN pip install --default-timeout=1200 nvidia-cublas-cu11
#RUN pip install --default-timeout=1200 nvidia-cuda-cupti-cu11
#RUN pip install --default-timeout=1200 nvidia-cuda-nvrtc-cu11
#RUN pip install --default-timeout=1200 nvidia-cuda-runtime-cu11
#RUN pip install --default-timeout=1200 nvidia-cudnn-cu11
#RUN pip install --default-timeout=1200 nvidia-cufft-cu11
#RUN pip install --default-timeout=1200 nvidia-curand-cu11
#RUN pip install --default-timeout=1200 nvidia-cusolver-cu11
#RUN pip install --default-timeout=1200 nvidia-nccl-cu11
#RUN pip install --default-timeout=1200 nvidia-nvtx-cu11
#RUN pip install --default-timeout=1200 -r requirements.txt

#CMD ["python3", "fed_server_run.py ${aggregation} ${clustering} ${splitting} ${model} ${dataset} ${offload} ${datasetlink} ${modellink}"]
RUN pip install --default-timeout=1200 certifi
RUN pip install --default-timeout=1200 charset-normalizer
RUN pip install --default-timeout=1200 cmake
RUN pip install --default-timeout=1200 exceptiongroup
RUN pip install --default-timeout=1200 filelock
RUN pip install --default-timeout=1200 idna
RUN pip install --default-timeout=1200 iniconfig
RUN pip install --default-timeout=1200 Jinja2
RUN pip install --default-timeout=1200 lit
RUN pip install --default-timeout=1200 MarkupSafe
RUN pip install --default-timeout=1200 mpmath
RUN pip install --default-timeout=1200 networkx
RUN pip install --default-timeout=1200 nn
RUN pip install --default-timeout=1200 numpy
RUN pip install --default-timeout=1200 packaging
RUN pip install --default-timeout=1200 Pillow
RUN pip install --default-timeout=1200 pluggy
RUN pip install --default-timeout=1200 pytest
RUN pip install --default-timeout=1200 requests
RUN pip install --default-timeout=1200 sympy
RUN pip install --default-timeout=1200 tomli
RUN pip install --default-timeout=1200 torch
RUN pip install --default-timeout=1200 torchvision
RUN pip install --default-timeout=1200 tqdm
RUN pip install --default-timeout=1200 triton
RUN pip install --default-timeout=1200 typing_extensions
RUN pip install --default-timeout=1200 urllib3
RUN pip install --default-timeout=1200 utils
RUN pip install --default-timeout=1200 vision

COPY app /fed-flow/app

WORKDIR /fed-flow/app/fl_training/runner/
ENTRYPOINT ["python3", "fed_client_run.py"]
