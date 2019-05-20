# printed-hw-segmentation

[![Codefresh build status]( https://g.codefresh.io/api/badges/pipeline/jumpst3r/Jumpst3r%2FBscThesis%2FBuildTestPush?branch=production&key=eyJhbGciOiJIUzI1NiJ9.NWNhYTQwZDAyYTE1MmZmMGQ2Y2FjOGM1.t3CzjCcStPDcqAcTi1nh8zpYB_E3tQmnemqSgDTbyQM&type=cf-1)]( https://g.codefresh.io/pipelines/BuildTestPush/builds?repoOwner=Jumpst3r&repoName=printed-hw-segmentation&serviceName=Jumpst3r%2Fprinted-hw-segmentation&filter=trigger:build~Build;branch:production;pipeline:5caa428088545f2b9e9e45e9~BuildTestPush) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![GitHub release](https://img.shields.io/github/release/jumpst3r/printed-hw-segmentation.svg)



## Introduction

**printed-hw-segmentation** is a tool that allows segmentation of printed and handwritten text using a fully convolutional network with CRF post-processing.

With each release a docker image of the code is published on [docker hub](https://cloud.docker.com/repository/docker/jumpst3r/printed-hw-segmentation). This image can be used in conjuction with [DIVA services](https://github.com/lunactic/DIVAServices) to provide segmenation as a web service. It can also be used locally.

## Local Usage

After pulling (`docker pull jumpst3r/printed-hw-segmentation:latest`) the image, the model can be applied to an image with 

`docker run -it --rm -v /FULL_PATH_TO/example.png:/input/example.png -v /FULL_PATH_TO_OUTPUT_FOLDER/:/output/ jumpst3r/printed-hw-segmentation sh /input/script.sh /input/example.png /output/`

The resulting image will be saved in the output path provided by the user

## Usage using the DIVA services framework

Follow the guide [here](https://lunactic.github.io/DIVAServicesweb/articles/installation/) to install the DIVA services framework on a server.

Once install, our method can be installed by make a POST request to the `alogithms` endpoint. This request should contain the JSON file `install.json` located in the `requests` folder of this repository. The JSON file `upload.json` can then be run to upload an image to the server. Finally the method can be executed using the JSON file `run.json` 
