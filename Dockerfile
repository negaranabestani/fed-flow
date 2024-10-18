FROM python:3.10

WORKDIR /fed-flow

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . .

WORKDIR /fed-flow/app/fl_training/runner
CMD ["python3", "fed_base_run.py"]