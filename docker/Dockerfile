FROM ubuntu:cosmic
LABEL maintainer="nicolas.dutly@unifr.ch"
RUN apt-get update && apt-get install libxcb1 libexpat1 dos2unix libgl1-mesa-glx --no-install-recommends -y && rm -rf /var/lib/apt/lists/*
COPY ./code/* /input/
ADD https://github.com/Jumpst3r/printed-hw-segmentation/releases/download/v1.0/printed-hw-segmentation /input/
RUN cd /input && chmod 777 printed-hw-segmentation && chmod +x script.sh && dos2unix script.sh && mkdir models && mv fcnn_bin.h5 models/
WORKDIR /input/
