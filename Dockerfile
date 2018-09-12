FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# install base tools
# Install some dependencies
RUN apt-get update && apt-get install -y \
		build-essential \
		unzip \
		swig \
		wget \
		pkg-config \
		zip \
		g++ \
		zlib1g-dev \
		cmake \
		curl \
		dh-autoreconf \
		vim \
		git \
		gfortran \
		libfreetype6-dev \
		libxft-dev \
		libncurses-dev \
		libopenblas-dev \
		libblas-dev \
		liblapack-dev \
		libatlas-base-dev \
		libcurl3-dev \
		python-dev \
		python-pydot \
		python-matplotlib \
		python-pip \
		python-numpy \
		python-pandas \
		python-sklearn \
		linux-headers-generic \
		linux-image-extra-virtual 

# Install SINGA from source
RUN git clone https://github.com/apache/incubator-singa.git /root/incubator-singa && \
    cd /root/incubator-singa && \
    git checkout f8cd7e3846a0eb016f2d511f826c18730eeda4cc && \
    mkdir build && \
    cd build && \
    cmake -DUSE_CUDA=ON -DUSE_MODULES=ON .. && \
    make && \
    cd python && \
    pip install .


# create directory
RUN mkdir /root/params

WORKDIR /root

# Copy files required for the app to run
COPY app.py /root
COPY admins.py /root
COPY errors.py /root
COPY users.py /root

# Copy files for inference
COPY model_bone_age.py /root
ADD bone_age /root/bone_age

# Tell the port number the container should expose
EXPOSE 5000

RUN pip install --upgrade pip
RUN pip install flask
RUN pip install flask-httpauth
RUN pip install flask-sqlalchemy
RUN pip install passlib




