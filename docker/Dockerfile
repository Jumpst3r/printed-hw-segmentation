FROM ubuntu:cosmic
LABEL maintainer="nicolas.dutly@unifr.ch"
RUN apt-get update && apt-get install libxcb1 libexpat1 dos2unix --no-install-recommends -y && rm -rf /var/lib/apt/lists/*
COPY ./code/* /input/
ADD https://github.com/Jumpst3r/printed-hw-segmentation/releases/download/v1.0/printed-hw-seg.bin /input/
RUN cd /input && chmod +x printed-hw-seg.bin && chmod +x script.sh && dos2unix script.sh
WORKDIR /input/