FROM nvidia/cuda:10.2-cudnn7-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.7 \
    python-pip

ADD requirements.txt requirements.txt
RUN pip install -U pip && pip install -r requirements.txt
