# syntax=docker/dockerfile:1.4

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG TARGET=cpu
ARG PYTHON_VERSION=3.11
ARG OMERO_VERSION=5.21.0
ARG TESTED=1

ENV TESTED=${TESTED}
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_BIN=python${PYTHON_VERSION}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-distutils \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-tk \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /workspace && chmod 777 /workspace \
    && mkdir -p /Biom3d && chmod 777 //Biom3d \
    #
    # Upgrade pip & install OMERO
    && ${PYTHON_BIN} -m pip install --upgrade pip setuptools wheel && \
    ${PYTHON_BIN} -m pip install \
    https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/download/20240202/zeroc_ice-3.6.5-cp311-cp311-manylinux_2_28_x86_64.whl \
    omero-py==${OMERO_VERSION}\
    && ${PYTHON_BIN} -m pip install --no-cache-dir --no-deps ezomero 
    
# Clone project
COPY . /Biom3d
WORKDIR /Biom3d

# Copy entrypoint
COPY entrypoint.sh /biom3d
RUN chmod +x /biom3d \
    #
    # Install biom3d
    && ${PYTHON_BIN} -m pip install . \
    #
    # Conditional: Install torch or fix symlink depending on CPU or GPU
    && if [ "$TARGET" = "cpu" ]; then \
        ${PYTHON_BIN} -m pip install torch --index-url https://download.pytorch.org/whl/cpu ; \
        elif [ "$TARGET" = "gpu" ]; then \
        ln -s /opt/conda/lib/libnvrtc.so.11.2 /opt/conda/lib/libnvrtc.so || true ; \
        fi

WORKDIR /workspace
ENTRYPOINT ["/biom3d"]