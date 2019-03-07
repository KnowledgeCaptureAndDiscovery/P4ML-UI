FROM registry.datadrivendiscovery.org/jpl/docker_images/jpl_base_python2.7
MAINTAINER D3M

RUN apt-get install -y libssl-dev

RUN \
pip install punk && \
pip install langdetect && \
pip install zerorpc && \
pip install stopit && \
pip install typing && \
pip install grpcio grpcio-tools && \
pip install librosa cvxpy h5py && \
pip install fancyimpute

# TA2 Server port
EXPOSE 45042

# Hack for Keras not to download stuff
COPY keras /root/.keras
ADD https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5 /root/.keras/models/

# Copy the TA2 code to the docker container
ENV CODE /dsbox
ENV TA2 $CODE/dsbox-ta2/python
ENV TA2API $CODE/ta3ta2-api
COPY . $CODE

# Compile TA3-TA2 api protocol files
RUN python -m grpc_tools.protoc -I $TA2API --python_out=$TA2API --grpc_python_out=$TA2API $TA2API/core.proto
RUN python -m grpc_tools.protoc -I $TA2API --python_out=$TA2API --grpc_python_out=$TA2API $TA2API/data_ext.proto
RUN python -m grpc_tools.protoc -I $TA2API --python_out=$TA2API --grpc_python_out=$TA2API $TA2API/dataflow_ext.proto

# Make a soft link to the search executable
RUN chmod 755 $TA2/ta2-search
RUN ln -s $TA2/ta2-search /usr/bin/ta2_search

# Start the TA2 server
CMD python $TA2/server/ta2-server.py
