#!/bin/bash

PYTHON_VERSION="3.11"
TORCH_VERSION="2.3.1"
OMERO_VERSION="5.21.0"
BIOM3D_VERSION="0.0.30"
ARCHITECTURE=x86_64
DOCKERFILE=template.dockerfile


# CPU
docker build \
  --build-arg BASE_IMAGE=ubuntu:22.04 \
  --build-arg TARGET=cpu \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  --build-arg OMERO_VERSION=$OMERO_VERSION \
  -t biom3d:${BIOM3D_VERSION}-$ARCHITECTURE-torch${TORCH_VERSION}-cpu \
  -f $DOCKERFILE .

# GPU
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime \
  --build-arg TARGET=gpu \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  --build-arg OMERO_VERSION=$OMERO_VERSION \
  -t biom3d:${BIOM3D_VERSION}-$ARCHITECTURE-torch2.3.1-cuda11.8-cudnn8 \
  -f $DOCKERFILE .

docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime \
  --build-arg TARGET=gpu \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  --build-arg OMERO_VERSION=$OMERO_VERSION \
  -t biom3d:${BIOM3D_VERSION}-$ARCHITECTURE-torch2.7.1-cuda11.8-cudnn9 \
  -f $DOCKERFILE .

docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime \
  --build-arg TARGET=gpu \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  --build-arg OMERO_VERSION=$OMERO_VERSION \
  --build-arg TESTED=0 \
  -t biom3d:${BIOM3D_VERSION}-$ARCHITECTURE-torch2.7.1-cuda12.8-cudnn9 \
  -f $DOCKERFILE .

docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime \
  --build-arg TARGET=gpu \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  --build-arg OMERO_VERSION=$OMERO_VERSION \
  --build-arg TESTED=0 \
  -t biom3d:${BIOM3D_VERSION}-$ARCHITECTURE-torch2.3.1-cuda12.1-cudnn8 \
  -f $DOCKERFILE .

