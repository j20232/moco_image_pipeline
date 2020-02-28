import unittest

from test_models import ModelTest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ModelTest))
    return suite
