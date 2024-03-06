FROM python:3.10
ADD requirements.txt /fed-flow/
ADD requirements1.txt /fed-flow/
ADD requirements2.txt /fed-flow/
ADD heavy-requirements.txt /fed-flow/
WORKDIR /fed-flow/
RUN pip install -r heavy-requirements.txt
RUN pip install -r requirements.txt
RUN pip install -r requirements1.txt
RUN pip install -r requirements2.txt
COPY app /fed-flow/app
COPY energy-estimation /fed-flow/energy
WORKDIR /fed-flow/