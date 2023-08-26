FROM python:3

RUN echo ${PWD}
RUN mkdir -p /fed-flow/app

COPY app /fed-flow/app
COPY requirements.txt /fed-flow/

WORKDIR /fed-flow/
RUN pip install --no-cache-dir -r requirements.txt

#CMD ["python3", "fed_server_run.py ${aggregation} ${clustering} ${splitting} ${model} ${dataset} ${offload} ${datasetlink} ${modellink}"]


WORKDIR /fed-flow/app/fl_training/runner/
ENTRYPOINT ["python3", "fed_edgeserver_run.py"]