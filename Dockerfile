FROM python:3.10.6
ADD requirements.txt /fed-flow/
COPY fedflow-env/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
WORKDIR /fed-flow/
#RUN pip install --default-timeout=1200 -r requirements.txt
COPY app /fed-flow/app