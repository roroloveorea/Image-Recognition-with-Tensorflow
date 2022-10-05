# docker build -t testing-f build_image.Dockerfile .

FROM nvcr.io/nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

ENV cwd="/home/"
WORKDIR $cwd

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# for old nvidia containers using the old signing key
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC F60F4B3D7FA2AF80

RUN apt-get -y update
# RUN apt-get install -y nvidia-docker2

RUN apt-get -y upgrade
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt -y update

RUN apt-get install -y \
    git 

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata python3-tk

# upgrade python to version 3.8 (IMPT: remove python3-dev and python3-pip if already installed)
RUN apt-get install -y python3.8-dev python3.8-venv python3-pip
RUN apt -y update
# Set python3.8 as the default python
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###
# RUN /usr/bin/python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade pip setuptools
RUN pip3 install tensorflow
RUN pip3 install matplotlib
RUN pip3 install keras
RUN pip3 install sklearn
RUN pip3 install tqdm
# COPY requirements.txt requirements.txt
# RUN pip3 install --no-cache-dir -r requirements.txt
# RUN mkdir /data
# WORKDIR /data
# COPY try.py /data/try.py
# RUN python3 try.py
#Required to save models in HDF5 format
RUN pip3 install pyyaml h5py  
RUN pip3 install tensorflow-datasets