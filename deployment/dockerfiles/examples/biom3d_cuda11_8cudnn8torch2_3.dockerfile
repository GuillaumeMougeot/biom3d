FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

#Install python and java dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    openjdk-11-jre-headless \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-distutils python3.11-venv python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /workspace && chmod 777 /workspace 
    
    #Install OMERO
RUN pip install numpy==1.26.4 --force-reinstall && \
    python3.11 -m pip install --upgrade pip setuptools wheel\
    && python3.11 -m pip install \
    https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/download/20240202/zeroc_ice-3.6.5-cp311-cp311-manylinux_2_28_x86_64.whl \
    omero-py==5.21.0\
    && ${PYTHON_BIN} -m pip install --no-cache-dir --no-deps ezomero

#Create entrypoint
COPY entrypoint.sh /biom3d
RUN chmod +x /biom3d\
    #
    # Install torch cpu and biom3d
    && python3.11 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python3.11 -m pip install --no-cache-dir biom3d\
    && ln -s /opt/conda/lib/libnvrtc.so.11.2 /opt/conda/lib/libnvrtc.so

# Volumes must be attached here
WORKDIR /workspace 
ENTRYPOINT ["biom3d"]
