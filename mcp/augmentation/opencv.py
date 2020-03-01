import numpy as np
import cv2


def get_random_kernel(min_ker, max_ker):
    structure = np.random.choice([cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS])
    kernel = cv2.getStructuringElement(structure, tuple(np.random.randint(min_ker, max_ker, 2)))
    return kernel


class RandomMorphology():
    def __init__(self, prob=0.4, min_ker=1, max_ker=5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.min_ker = min_ker
        self.max_ker = max_ker

    def __call__(self, img):
        if np.random.uniform() < self.prob:
            r = np.random.uniform()
            kernel = get_random_kernel(self.min_ker, self.max_ker)
            if r < 0.25:
                img = cv2.erode(img, kernel, iterations=1)
            elif r < 0.50:
                img = cv2.dilate(img, kernel, iterations=1)
            elif r < 0.75:
                k2 = get_random_kernel(self.min_ker, self.max_ker)
                img = cv2.erode(img, kernel, iterations=1)
                img = cv2.dilate(img, k2, iterations=1)
            else:
                k2 = get_random_kernel(self.min_ker, self.max_ker)
                img = cv2.dilate(img, kernel, iterations=1)
                img = cv2.erode(img, k2, iterations=1)
        return img
