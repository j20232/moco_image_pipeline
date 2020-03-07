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

# ----------------------------------- GridDistortion -----------------------------------------


class RandomGridDistortion():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image
        num_step = 5
        distort = self.magnitude

        distort_x = [1 + np.random.uniform(-distort, distort) for i in range(num_step + 1)]
        distort_y = [1 + np.random.uniform(-distort, distort) for i in range(num_step + 1)]
        height, width = image.shape[:2]

        xx = np.zeros(width, np.float32)
        step_x = width // num_step
        prev = 0
        for i, x in enumerate(range(0, width, step_x)):
            start = x
            end = x + step_x
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + step_x * distort_x[i]
            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        yy = np.zeros(height, np.float32)
        step_y = height // num_step
        prev = 0
        for idx, y in enumerate(range(0, height, step_y)):
            start = y
            end = y + step_y
            if end > height:
                end = height
                cur = height
            else:
                cur = prev + step_y * distort_y[idx]
            yy[start:end] = np.linspace(prev, cur, end - start)
            prev = cur
        map_x, map_y = np.meshgrid(xx, yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image


class RandomCustomGridDistortion():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image
        distort = self.magnitude * 0.3
        height = image.shape[0]
        width = image.shape[1]
        s_x = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        s_y = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
        d_x = s_x.copy()
        d_y = s_y.copy()
        d_x[[1, 4, 7]] += np.random.uniform(-distort, distort, 3)
        d_y[[3, 4, 5]] += np.random.uniform(-distort, distort, 3)
        s_x = (s_x * width)
        s_y = (s_y * height)
        d_x = (d_x * width)
        d_y = (d_y * height)
        distort = np.zeros_like(image)
        for index in ([4, 1, 3], [4, 1, 5], [4, 7, 3], [4, 7, 5]):
            point = np.stack([s_x[index], s_y[index]]).T
            qoint = np.stack([d_x[index], d_y[index]]).T
            src = np.array(point, np.float32)
            dst = np.array(qoint, np.float32)
            mat = cv2.getAffineTransform(src, dst)
            point = np.round(point).astype(np.int32)
            x0 = np.min(point[:, 0])
            x1 = np.max(point[:, 0])
            y0 = np.min(point[:, 1])
            y1 = np.max(point[:, 1])
            mask = np.zeros_like(image)
            mask[y0:y1, x0:x1] = 1
            mask = mask * image
            warp = cv2.warpAffine(mask, mat, (width, height), borderMode=cv2.BORDER_REPLICATE)
            distort = np.maximum(distort, warp)

        return distort

# ----------------------------------- Contrast -----------------------------------------


class RandomContrast():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image

        alpha = 1 + np.random.uniform(-1, 1) * self.magnitude
        image = image.astype(np.float32) * alpha
        image = np.clip(image, 0, 1)
        return image


class RandomBlockFade():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image

        size = [0.1, self.magnitude]
        height = image.shape[0]
        width = image.shape[1]

        # get bounding box
        m = image.copy()
        cv2.rectangle(m, (0, 0), (height, width), 1, 5)
        m = image < 0.5
        if m.sum() == 0:
            return image

        m = np.where(m)
        y0, y1, x0, x1 = np.min(m[0]), np.max(m[0]), np.min(m[1]), np.max(m[1])
        w = x1 - x0
        h = y1 - y0
        if w * h < 10:
            return image

        ew, eh = np.random.uniform(*size, 2)
        ew = int(ew * w)
        eh = int(eh * h)
        ex = np.random.randint(0, w - ew) + x0
        ey = np.random.randint(0, h - eh) + y0
        image[ey:ey + eh, ex:ex + ew] *= np.random.uniform(0.1, 0.5)
        image = np.clip(image, 0, 1)
        return image


# ------------------------------------ Noise ------------------------------------------


class RandomErode():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image
        s = int(round(1 + np.random.uniform(0, 1) * self.magnitude * 6))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s, s)))
        image = cv2.erode(image, kernel, iterations=1)
        return image


class RandomDilate():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image
        s = int(round(1 + np.random.uniform(0, 1) * self.magnitude * 6))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s, s)))
        image = cv2.dilate(image, kernel, iterations=1)
        return image


class RandomSpinkle():
    def __init__(self, prob, magnitude=0.5, size=16):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5
        self.size = size

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image
        num_sprinkle = int(round(1 + np.random.randint(10) * self.magnitude))
        image = image.copy()
        image_small = cv2.resize(image, dsize=None, fx=0.25, fy=0.25)
        m = np.where(image_small > 0.25)
        num = len(m[0])
        if num == 0:
            return image
        s = self.size // 2
        i = np.random.choice(num, num_sprinkle)
        for y, x in zip(m[0][i], m[1][i]):
            y = y * 4 + 2
            x = x * 4 + 2
            image[y - s:y + s, x - s:x + s] = 0
        return image


class RandomNoise():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image
        noise = np.random.uniform(-1, 1, image.shape) * self.magnitude * 0.7
        image = image + noise
        image = np.clip(image, 0, 1)
        return image


class RandomLine():
    def __init__(self, prob, magnitude=0.5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.magnitude = 0.5

    def __call__(self, image):
        if np.random.uniform() > self.prob:
            return image
        num_lines = int(round(1 + np.random.randint(8) * self.magnitude))
        height = image.shape[0]
        width = image.shape[1]
        image = image.copy()

        def line0():
            return (0, 0), (width - 1, 0)

        def line1():
            return (0, height - 1), (width - 1, height - 1)

        def line2():
            return (0, 0), (0, height - 1)

        def line3():
            return (width - 1, 0), (width - 1, height - 1)

        def line4():
            x0, x1 = np.random.choice(width, 2)
            return (x0, 0), (x1, height - 1)

        def line5():
            y0, y1 = np.random.choice(height, 2)
            return (0, y0), (width - 1, y1)

        for i in range(num_lines):
            p = np.array([1 / 4, 1 / 4 , 1 / 4, 1 / 4, 1, 1])
            func = np.random.choice([line0, line1, line2, line3, line4, line5], p=p / p.sum())
            (x0, y0), (x1, y1) = func()
            color = np.random.uniform(0, 1)
            thickness = np.random.randint(1, 5)
            line_type = np.random.choice([cv2.LINE_AA, cv2.LINE_4, cv2.LINE_8])
            cv2.line(image, (x0, y0), (x1, y1), color, thickness, line_type)

        return image
