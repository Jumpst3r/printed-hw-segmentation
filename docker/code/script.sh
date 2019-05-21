#!/bin/sh
inputImage=${1}
outputFolder=${2}

/input/printed-hw-segmentation --enableCRF ${inputImage} ${outputFolder}
