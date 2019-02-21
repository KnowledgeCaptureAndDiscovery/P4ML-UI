import unittest

def suite():
    test_suite = unittest.TestLoader().discover('.', pattern='*_test.py')
    return test_suite

if __name__=='__main__':
    unittest.run(suite())
