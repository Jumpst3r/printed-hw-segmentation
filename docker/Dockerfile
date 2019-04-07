FROM continuumio/miniconda3
LABEL maintainer="nicolas.dutly@unifr.ch"

RUN apt-get update
RUN apt-get install linux-headers-amd64 build-essentials
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

ADD script.sh /input/script.sh
ADD printed-hw-seg.py /input/printed-hw-seg.py
ADD fcn_helper_function.py /input/fcn_helper_function.py
ADD fcnn_bin.h5 /input/fcnn_bin.h5
ADD post_processing.py /input/post_processing.py
ADD img_utils.py /input/img_utils.py