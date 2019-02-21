'''
    spider.tests: featurization_vgg16_test.py

    @tsnowak

    Unit tests for the spider.featurization.vgg16 module
'''

import unittest
import tempfile
import os.path
import shutil
import numpy as np

from spider.featurization.vgg16 import VGG16

# disable TF warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class testVGG16(unittest.TestCase):


    def test_image_input(self):
        """ Test input_handler given image input.         
        """
       
        vgg16 = VGG16()
        data_dir = os.path.dirname(os.path.abspath(__file__))
        data = os.path.join(data_dir, 'data/elephant.jpg')
        output = vgg16.input_handler(data)

        self.assertEqual(len(output.shape), 4) 
        self.assertEqual(output.shape[1], 224)
        self.assertEqual(output.shape[2], 224)
        self.assertEqual(output.shape[3], 3)

    def test_nparray_input(self):
        """ Test input_handler given nparray input. 
        """

        vgg16 = VGG16()
        data = np.ones((224,224,3))
        output = vgg16.input_handler(data)

        self.assertEqual(len(output.shape), 4) 
        self.assertEqual(output.shape[1], 224)
        self.assertEqual(output.shape[2], 224)
        self.assertEqual(output.shape[3], 3)

        data = np.random.rand(120,75,3)
        output = vgg16.input_handler(data)

        self.assertEqual(len(output.shape), 4) 
        self.assertEqual(output.shape[1], 224)
        self.assertEqual(output.shape[2], 224)
        self.assertEqual(output.shape[3], 3)

    def test_model(self):
        """ Verify shapes of features at varying layers in the model
        """

        vgg16 = VGG16()
        data_dir = os.path.dirname(os.path.abspath(__file__))
        data = os.path.join(data_dir, 'data/elephant.jpg')

        features = vgg16.produce(data)
        self.assertEqual(len(features[-1]), 4096)

if __name__ == '__main__':
    unittest.main()
