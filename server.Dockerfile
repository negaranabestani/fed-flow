FROM python:3.10.6

RUN echo ${PWD}
RUN mkdir -p /fed-flow/app


ADD requirements.txt /fed-flow/
COPY fedflow-env/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
WORKDIR /fed-flow/
#RUN pip install --default-timeout=1200 -r requirements.txt
COPY app /fed-flow/app

#WORKDIR /fed-flow/app/rl_training/runner
#ENTRYPOINT ["python3", "fed_server_run.py"]