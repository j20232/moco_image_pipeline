FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ARG PYTHON_VERSION=3.7
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# This must be done before pip so that requirements.txt is available
WORKDIR /opt/pytorch
COPY . .

# pytorch
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

RUN git submodule update --init --recursive
WORKDIR /workspace
RUN chmod -R a+w .
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
