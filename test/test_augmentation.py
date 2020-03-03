import unittest

import numpy as np
from PIL import Image
from mcp import augmentation as maug


class AugmentationTest(unittest.TestCase):

    def test_tensor(self):
        float_img = np.random.rand(255, 255, 3)
        uint_img = (float_img * 255).astype(np.uint8)
        img = Image.fromarray(uint_img)
        size = [224, 224]
        for method_name in maug.pil_modules:
            module = getattr(maug, method_name)(size=size, prob=1.0)

            # np.float32
            out = module(float_img)
            self.assertTrue(out.shape[0] == size[0] and out.shape[1] == size[1],
                            "np.float32: illegal size")
            self.assertTrue(out.max() <= 1.0, "np.float32: max value <= 1.0")
            self.assertTrue(out.min() >= 0.0, "np.float32: min value >= 0.0")
# np.uint8 out = module(uint_img)
            self.assertTrue(out.shape[0] == size[0] and out.shape[1] == size[1],
                            "np.uint8: illegal size")
            self.assertTrue(out.max() <= 255, "np.uint8: max value <= 255")
            self.assertTrue(out.min() >= 0, "np.uint8: min value >= 0")

            # Image -> np.uint8
            out = module(img)
            self.assertTrue(out.shape[0] == size[0] and out.shape[1] == size[1],
                            "Image: illegal size")
            self.assertTrue(out.max() <= 255, "Image -> np.uint8: max value <= 255")
            self.assertTrue(out.min() >= 0, "Image -> np.uint8: min value >= 0")

    def test_augmentation(self):
        for method_name in maug.modules:
            print("testing", method_name)
            # float64 prob=0.0
            img = np.random.rand(255, 255, 3).astype(np.float64)
            module0 = getattr(maug, method_name)(prob=0.0)
            out = module0(img)
            self.assertTrue(img.shape == out.shape, "Prob=0.0: inconsistent shape")
            self.assertTrue(out.max() <= 1.0, "Prob=0.0: max value <= 1.0")
            self.assertTrue(out.min() >= 0.0, "Prob=0.0: min value >= 0.0")

            # float64 prob=1.0
            module1 = getattr(maug, method_name)(prob=1.0)
            out = module1(img)
            self.assertTrue(img.shape == out.shape, "Prob=1.0: inconsistent shape")
            self.assertTrue(out.max() <= 1.0, "Prob=1.0: max value <= 1.0")
            self.assertTrue(out.min() >= 0.0, "Prob=1.0: min value >= 0.0")

