FROM python:3.10
USER root
#RUN apt upgrade
#RUN apt install -y python3
#RUN apk add --no-cache bash
ADD requirements.txt /fed-flow/
COPY fedflow-env/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY app /fed-flow/app
COPY energy-estimation /fed-flow/energy

#WORKDIR /fed-flow/
#RUN pip install --default-timeout=1200 -r requirements.txt

#COPY comm.sh /fed-flow/
#COPY energy/system_utils.py /fed-flow/
#WORKDIR /fed-flow/
#RUN apk add wireless-tools
#ENTRYPOINT ["/bin/sh"]
#CMD ["comm.sh"]
