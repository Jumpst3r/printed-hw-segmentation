FROM ubuntu:latest
LABEL maintainer="nicolas.dutly@unifr.ch"
RUN apt-get update && apt-get install libxcb1 --no-install-recommends -y && rm -rf /var/lib/apt/lists/*
COPY ./code/* /input/
ADD https://github.com/Jumpst3r/printed-hw-segmentation/releases/download/v0.1/printed-hw-seg.bin /input/
WORKDIR /input/