#FROM registry.datadrivendiscovery.org/jpl/docker_images:python3.6_base
FROM python:3.6-stretch
MAINTAINER D3M

RUN apt-get update
RUN apt-get -y install libssl-dev 
#python3-dev python3-setuptools
RUN apt-get -y install ffmpeg flac

RUN \
pip3 install --upgrade pip && \
pip3 install punk && \
pip3 install langdetect && \
pip3 install zerorpc && \
pip3 install stopit && \
pip3 install grpcio grpcio-tools && \
pip3 install librosa cvxpy h5py && \
pip3 install fancyimpute

# More python packages
RUN pip3 install pandas keras tensorflow
RUN pip3 install pillow
RUN pip3 install pslgraph networkx

# Install D3M stuff
RUN pip3 install --process-dependency-links primitive-interfaces

EXPOSE 45042

# Hack for Keras not to download stuff
COPY keras /root/.keras
ADD https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5 /root/.keras/models/

# Copy the TA2 code to the docker container
ENV CODE /dsbox
ENV TA2 $CODE/dsbox-ta2/python
ENV TA2API $CODE/ta3ta2-api
COPY . $CODE

# PIP install d3m types
#RUN cd $CODE/types && pip3 install .

# PYTHON 3 changes
# Change environment from python to python3 in code
RUN \
sed -i.bak 's/env python/env python3/' \
$TA2/dsbox/executer/executionhelper.py \
$TA2/ta2-search \
$TA2/server/ta2-server.py \
$TA2/server/ta2-client.py

# Change primitives unified interface to use python3
RUN \
sed -i.bak 's/primitive-interfaces\/python2/primitive-interfaces/' \
$TA2/dsbox_dev_setup.py

# Compile TA3-TA2 api protocol files
RUN python3 -m grpc_tools.protoc -I $TA2API --python_out=$TA2API --grpc_python_out=$TA2API $TA2API/core.proto
RUN python3 -m grpc_tools.protoc -I $TA2API --python_out=$TA2API --grpc_python_out=$TA2API $TA2API/data_ext.proto
RUN python3 -m grpc_tools.protoc -I $TA2API --python_out=$TA2API --grpc_python_out=$TA2API $TA2API/dataflow_ext.proto

# Make a soft link to the search executable
RUN chmod 755 $TA2/ta2-search
RUN ln -s $TA2/ta2-search /usr/bin/ta2_search

# Start the TA2 server
CMD python3 $TA2/server/ta2-server.py
