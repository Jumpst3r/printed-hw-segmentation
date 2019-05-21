#!/bin/sh
inputImage=${1}
outputFolder=${2}

/input/printed-hw-segmentation --enableCRF ${inputImage} ${outputFolder}
#remove non CRF version (ugly fix)
rm /output/fcn_out.png
