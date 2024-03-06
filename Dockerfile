FROM python:3.10
ADD requirements.txt /fed-flow/
ADD heavy-requirements.txt /fed-flow/
RUN pip install -r requirements.txt
RUN pip install -r heavy-requirements.txt
COPY app /fed-flow/app
COPY energy-estimation /fed-flow/energy
WORKDIR /fed-flow/