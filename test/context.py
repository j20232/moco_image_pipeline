import unittest

from test_models import ModelTest
from test_augmentation import AugmentationTest


def suite():
    suite = unittest.TestSuite()
    # suite.addTest(unittest.makeSuite(ModelTest))
    suite.addTest(unittest.makeSuite(AugmentationTest))
    return suite
