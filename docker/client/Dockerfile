FROM python:3

WORKDIR ../../
ADD requirements.txt /

RUN pip install -r requirements.txt

#CMD ["python3", "fed_server_run.py ${aggregation} ${clustering} ${splitting} ${model} ${dataset} ${offload} ${datasetlink} ${modellink}"]
COPY app/ /app/

WORKDIR app/fl_training/runner/
ENTRYPOINT ["python3", "fed_client_run.py"]