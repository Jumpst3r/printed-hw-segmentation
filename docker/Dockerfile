FROM continuumio/miniconda3
LABEL maintainer="nicolas.dutly@unifr.ch"

COPY ./code/* /input/

RUN apt-get update && apt-get install build-essential -y --no-install-recommends && rm -rf /var/lib/apt/lists/*
COPY ./code/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
# Pull the environment name out of the environment.yml
RUN echo "source activate productionDocker" > ~/.bashrc
ENV PATH /opt/conda/envs/productionDocker/bin:$PATH

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
ENV CONDA_ENV_PATH /opt/miniconda
ENV CONDA_ACTIVATE "source $CONDA_ENV_PATH/bin/activate productionDocker"
