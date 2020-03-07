import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pathlib
from pathlib import Path

import os
import sys
sys.path.append(os.path.join("."))
import mcp.augmentation as maug


def show_imgs(img_list, title_list=None,
              rows=1, cmap="viridis", size=(16, 8), show_axis=False):
    if title_list is not None:
        assert len(img_list) == len(title_list)
    if type(img_list[0]) == pathlib.PosixPath or type(img_list[0]) == str:
        show_list = [Image.open(str(img_path)) for img_path in img_list]
    else:
        show_list = img_list
    plt.figure(figsize=size)
    assert len(show_list) % rows == 0
    for idx, img in enumerate(show_list):
        plt.subplot(rows, int(len(show_list) / rows), idx + 1)
        plt.imshow(img, cmap=cmap)
        if title_list is not None:
            plt.title(title_list[idx])
        if not show_axis:
            plt.axis("off")
    plt.show()


if __name__ == "__main__":
    img_nums = 100
    rows = 10
    TEST_PATH = Path(".").resolve() / "assets" / "test_images"
    img = cv2.imread(str(TEST_PATH / "3.png"))
    img = cv2.resize(img, (128, 128))
    img = (img / 255).astype(np.float64)
    # augmentation
    imgs = []
    for _ in range(img_nums):
        out = img
        for method_name in maug.modules:
            module = getattr(maug, method_name)(prob=0.2)
            out = module(out)
        tmp = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        imgs.append(tmp)
    show_imgs(imgs, rows=rows, size=(24, 10))
