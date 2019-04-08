FROM continuumio/miniconda3
LABEL maintainer="nicolas.dutly@unifr.ch"

COPY ./code/* /input/

RUN apt-get update && apt-get install build-essential -y --no-install-recommends && rm -rf /var/lib/apt/lists/*
COPY ./code/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH