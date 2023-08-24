FROM python:3

#WORKDIR ../../../
#ADD ../../../requirements.txt /
RUN echo ${PWD} && ls -lR
RUN mkdir -p /fed-flow/app
COPY ../../app /fed-flow/app/
COPY ../../requirements.txt /fed-flow/
#ADD ../../requirements.txt /
#ADD ../../app /
WORKDIR /fed-flow/
RUN pip install -r requirements.txt

#CMD ["python3", "fed_server_run.py ${aggregation} ${clustering} ${splitting} ${model} ${dataset} ${offload} ${datasetlink} ${modellink}"]


WORKDIR /fed-flow/app/fl_training/runner/
ENTRYPOINT ["python3", "fed_server_run.py"]
