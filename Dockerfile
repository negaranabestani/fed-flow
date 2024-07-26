FROM autlpdslab/fedflow:base
COPY app /fed-flow/app
COPY energy-estimation /fed-flow/energy
WORKDIR /fed-flow/
COPY requirements4.txt .
RUN pip install -r requirements4.txt