# Reference: https://www.kaggle.com/c/bengaliai-cv19/discussion/123757

import numpy as np
import cv2


# ----------------------------------- Geometric -----------------------------------------

class RandomProjective():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image

        mag = np.random.uniform(-1, 1) * 0.5 * self.magnitude
        height, width = image.shape[:2]
        x0, y0 = 0, 0
        x1, y1 = 1, 0
        x2, y2 = 1, 1
        x3, y3 = 0, 1

        mode = np.random.choice(['top', 'bottom', 'left', 'right'])
        if mode == 'top':
            x0 += mag
            x1 -= mag
        if mode == 'bottom':
            x3 += mag
            x2 -= mag
        if mode == 'left':
            y0 += mag
            y3 -= mag
        if mode == 'right':
            y1 += mag
            y2 -= mag
        s = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * [[width, height]]
        d = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]]) * [[width, height]]
        transform = cv2.getPerspectiveTransform(s.astype(np.float32), d.astype(np.float32))
        image = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image


class RandomPerspective():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image

        mag = np.random.uniform(-1, 1, (4, 2)) * 0.25 * self.magnitude
        height, width = image.shape[:2]
        s = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        d = s + mag
        s *= [[width, height]]
        d *= [[width, height]]
        transform = cv2.getPerspectiveTransform(s.astype(np.float32), d.astype(np.float32))
        image = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image


class RandomRotate():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image

        angle = 1 + np.random.uniform(-1, 1) * 30 * self.magnitude
        height, width = image.shape[:2]
        cx, cy = width // 2, height // 2
        transform = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image


class RandomScale():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image

        s = 1 + np.random.uniform(-1, 1) * self.magnitude * 0.5
        height, width = image.shape[:2]
        transform = np.array([[s, 0, 0], [0, s, 0], ], np.float32)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image


class RandomShearX():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image

        sx = np.random.uniform(-1, 1) * self.magnitude
        height, width = image.shape[:2]
        transform = np.array([[1, sx, 0], [0, 1, 0]], np.float32)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image


class RandomShearY():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image

        sy = np.random.uniform(-1, 1) * self.magnitude
        height, width = image.shape[:2]
        transform = np.array([[1, 0, 0], [sy, 1, 0]], np.float32)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image


class RandomStretchX():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image

        sx = 1 + np.random.uniform(-1, 1) * self.magnitude
        height, width = image.shape[:2]
        transform = np.array([[sx, 0, 0], [0, 1, 0]], np.float32)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image


class RandomStretchY():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image

        sy = 1 + np.random.uniform(-1, 1) * self.magnitude
        height, width = image.shape[:2]
        transform = np.array([[1, 0, 0], [0, sy, 0]], np.float32)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image


