FROM python:3.10.9

WORKDIR /fed-flow


ADD app /fed-flow/app/


COPY requirements.txt /fed-flow
RUN pip install -r requirements.txt

#CMD ["python3", "fed_server_run.py ${aggregation} ${clustering} ${splitting} ${model} ${dataset} ${offload} ${datasetlink} ${modellink}"]


#WORKDIR /fed-flow/app/fl_training/runner/
#ENTRYPOINT ["python3", "fed_server_run.py"]
