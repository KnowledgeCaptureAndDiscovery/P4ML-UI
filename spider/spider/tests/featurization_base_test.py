'''
    spider.tests: test_featurization_base.py

    jason corso

    Unit tests for the spider.featurization.base module
'''

import unittest
import tempfile
import os.path
import shutil

class testBasicMethods(unittest.TestCase):

    def test_count(self):
        size = 3
        self.assertEquals(size,3)

if __name__ == '__main__':
    unittest.main()
