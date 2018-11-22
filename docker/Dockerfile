FROM nvidia/cuda:9.0-cudnn7-devel

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      bzip2 \
      build-essential \
      gcc-6 \
      g++-6 \
      git \
      cmake \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

# Install conda and related packages
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

ARG python_version=3.6

RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install \
      sklearn_pandas && \
    conda install \
      bcolz \
      h5py \
      matplotlib \
      mkl \
      nose \
      notebook \
      Pillow \
      pandas \
      pydot \
      pygpu \
      pyyaml \
      scikit-learn \
      six && \
    conda clean -yt

# Install oytorch
RUN conda install torchvision pytorch=0.4.1 -c pytorch

# Install Sean Naren's warp-ctc
RUN git clone https://github.com/SeanNaren/warp-ctc.git
RUN cd warp-ctc; mkdir build; cd build; cmake ..; make
RUN cd warp-ctc; cd pytorch_binding; python setup.py install

# We need gcc-6 and g++-6 to compile nms
RUN rm -rf /usr/bin/gcc && rm -rf /usr/bin/g++ && \
    ln -s /usr/bin/gcc-6 /usr/bin/gcc && \
    ln -s /usr/bin/g++-6 /usr/bin/g++

# Install opencv3 Python bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgtk2.0-0 \
      libcanberra-gtk-module && \
    rm -rf /var/lib/apt/lists/*

RUN conda install -y -c menpo opencv3=3.1.0 && \
    conda clean -ya

WORKDIR /workspace
