Using Docker for Reproducible E2E-MLT Build Environment
=======================================================

## Prerequisite

The following procedures are tested on machine with

* Ubuntu 16.04.5 LTS
* Docker version 17.12.0-ce, build c97c6d6
* CUDA Version 8.0.61
* CUDNN 7

You'll also need to install [`nvidia-docker2`](https://github.com/NVIDIA/nvidia-docker).

## Instructions for running the demo

1. Inside this directory, run `. ./build_docker.sh` to build the docker image. The image will take up approximately 8GB.
2. After the image is built, first edit `PRETRAINED_MODEL_PATH` in `run_docker.sh` to the directory that contains the pretrained model `e2e-mlt.h5`. Next, run `. ./run_docker.sh`.
3. By default, you'll be inside `/workspace` directory, which contains all the files inside the E2E-MLT repo.
