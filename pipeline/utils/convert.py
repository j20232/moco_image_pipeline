import numpy as np
import cv2


def calc_bounding_box(img):
    cols = np.any(img, axis=0)
    xmin, xmax = np.where(cols)[0][[0, -1]]
    rows = np.any(img, axis=1)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    return xmin, xmax, ymin, ymax


def crop_and_resize_img(img, size, max_width, max_height,
                        crop_width=13, crop_height=10,
                        padding=16, line_threshold=80, noise_threshold=28):
    # crop a box around pixels larger than line_threshold
    # NOTE: some images contain line at the sides
    xmin, xmax, ymin, ymax = calc_bounding_box(img[5:-5, 5:-5] > 80)

    # cropping may cut too much, so we need to add it back
    xmin = xmin - crop_width if (xmin > crop_width) else 0
    ymin = ymin - crop_height if (ymin > crop_height) else 0
    xmax = xmax + crop_width if (xmax < max_width - crop_width) else max_width
    ymax = ymax + crop_height if (ymax < max_height - crop_height) else max_height
    cropped_img = img[ymin:ymax, xmin:xmax]

    # remove low intensity pixels as noise
    cropped_img[cropped_img < noise_threshold] = 0

    # make sure that aspect ratio is kept in rescaling
    len_x, len_y = xmax - xmin, ymax - ymin
    length = max(len_x, len_y) + padding
    out_img = np.pad(cropped_img, [((length - len_y) // 2,), ((length - len_x) // 2,)], mode="constant")
    return cv2.resize(out_img, (size, size))
