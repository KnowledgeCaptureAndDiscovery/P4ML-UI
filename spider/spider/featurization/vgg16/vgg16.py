from __future__ import print_function

import sys
import os.path
import inspect
import urllib
import warnings
import numpy as np
from scipy import misc

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

from . import utils

from typing import *
from spider.featurization_base import FeaturizationTransformerPrimitiveBase

# Params could include option of weights to load into model, but currently
# model is fixed and no constraints change instance of primitive

Inputs = list
Outputs = list

class VGG16(FeaturizationTransformerPrimitiveBase[Inputs, Outputs]):


    def __init__(
            self,
            device=('cpu', 1),
            num_cores=4,
            output_feature_layer='fc1'):
        """
        Darpa D3M VGG16 Image Featurization Primitive

        Arguments:
            - device: Device type ('cpu' or 'gpu') and number of devices as a tuple.
            - num_cores: Integer number of CPU cores to use.
            - output_feature_layer: String layer name whose features to output.

        Return :
            - None
        """

        self.output_feature_layer = output_feature_layer

        # settable vars unrelated to fit/application of primitive
        self.device=device
        self.num_cores=num_cores

        if self.device[0].lower() == 'cpu':
            num_GPU = 0
            num_CPU = self.device[1]
        elif self.device[0].lower() == 'gpu':
            num_CPU = 1
            num_GPU = self.device[1]
        else:
            raise ValueError('Invalid device tuple supplied. Options: (\'cpu\', <num_cpu>), '\
                '(\'gpu\', <num_gpu>).')

        config = tf.ConfigProto(intra_op_parallelism_threads=self.num_cores,\
            inter_op_parallelism_threads=self.num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
        session = tf.Session(config=config)
        K.set_session(session)

        # helper vars
        self.input_shape = (224, 224, 3)
        self.weights_url = 'https://umich.box.com/shared/static/dzmxth5l7ql3xggc0hst5h1necjfaurt.h5'
        self.weights_directory = os.path.join(os.path.abspath(os.path.dirname(utils.__file__)), \
            'weights')
        if not os.path.isdir(self.weights_directory):
            os.makedirs(self.weights_directory)
        self.weights_filename = 'vgg16_weights.h5'
        self.interpolation_method = 'bilinear'
        self.base_model = self.model()

        # output variable of list of feature vectors
        self.features = []

    def model(self):

        """
        Implementation of the VGG16 architecture provided by keras and
            fchollet on github with minor alterations. This implementation is
            designed to be used with a Tensorflow backend. As such, ensure that
            the pixel depth channel is last.

        Arguments:
            - None

        Returns:
            - A Keras model instance.
        """

        # Prepare image for input to model
        img_input = Input(shape=self.input_shape)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

        inputs = img_input

        # Create model.
        model = Model(inputs, x, name='vgg16')

        model.load_weights(utils.download_weights(self.weights_directory,\
                self.weights_filename, self.weights_url))
    
        return model

    def input_handler(self, data, method='bilinear'):
        """
        Performs necessary manipulations to prepare input
            data for implementation in the VGG16 network

        Arguments:
            - data: string file location of image file, or np.array of an image
                to feed into the network
            - method: optional -- default 'bilinear' --, selects which method of
                interpolation to perform when resizing the input is necessary

        Returns:
            - x: 4D np.array to feed into the VGG16 network. Typically the output
                is always of shape (1, 224, 224, 3)
        """

        # interpolation methods handling
        method_list = {'bilinear', 'nearest', 'lanczos', 'bicubic', 'cubic'}
        if method not in method_list:
            raise ValueError('Method for interpolation must be one of the following %s' \
                % str(method_list))

        # handle if input is a path
        if isinstance(data, str):
            if not os.path.isfile(data):
                raise IOError('Input data file does not exist.')
            else:
                img = image.load_img(data, target_size=(self.input_shape[0],
                    self.input_shape[1]))
                x = image.img_to_array(img)
                if len(x.shape) is 3:
                    x = np.expand_dims(x, axis=0)

        # handle if input is an np.ndarray
        elif isinstance(data, np.ndarray):
            x = utils.interpolate(data, self.input_shape, method)
            if len(x.shape) is 3:
                x = np.expand_dims(x, axis=0)

        # if neither raise error
        else:
            raise TypeError('Input must either be a file path to an image, \
                    or an np.ndarray of an image.')

        assert x.shape == (1,) + self.input_shape

        x = preprocess_input(x)

        return x

    def model_handler(self, output_feature_layer):
        """
        Used to construct and layout the VGG16 model with desired input and output

        Arguments: 
            - output_feature_layer: The layer from VGG16 to output from the forward pass
                of the model

        Returns:
            - A Keras model type with desired input and output
        """ 

        # allow for layer feature selection, but defaulting to output of conv-layers to make
        # dataset agnostic -- intend for SVM to be run after
        try:
            model = Model(inputs=self.base_model.input, outputs=\
                    self.base_model.get_layer(output_feature_layer).output)
        except AttributeError:
            warnings.warn('Improper layer output selected, defaulting to \'fc1\' layer features.')
            model = Model(inputs=self.base_model.input,
                    outputs=self.base_model.get_layer('fc1').output)

        return model


    def imagenet_predictions(self, data):
        """
        Outputs the predicted imagenet class of the input data
        
        Arguments:
            - data: file location or numpy array containing image data to feed into the network

        Returns:
            - predictions: The top 5 predicted imagenet classes at the output of the model
        """

        model = self.model_handler('predictions')
        x = self.input_handler(data, self._interpolation_method)
        features = model.predict(x)
        predictions = decode_predictions(features)

        return predictions

    def produce(self, inputs, timeout=None, iterations=None):

        model = self.model_handler(self.output_feature_layer)

        if isinstance(inputs, list):
            for datum in inputs:
                x = self.input_handler(datum, self.interpolation_method)
                image_features = model.predict(x)
                image_features = np.squeeze(image_features)
                self.features.append(image_features)
        else:
                x = self.input_handler(inputs, self.interpolation_method)
                image_features = model.predict(x)
                image_features = np.squeeze(image_features)
                self.features.append(image_features)

        return self.features
