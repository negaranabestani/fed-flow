FROM autlpdslab/fedflow:base
COPY app /fed-flow/app
COPY energy-estimation /fed-flow/energy
WORKDIR /fed-flow/
RUN pip install kombu
