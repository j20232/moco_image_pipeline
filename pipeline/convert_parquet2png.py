import os
import zipfile
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from utils.convert import crop_and_resize_img


# Please set the following params
INPUT_DIR = Path(".").resolve() / "input"
TRAIN_DIR = INPUT_DIR / "train_images"
TRAIN_ZIPFILES = ["train_image_data_0.parquet.zip",
                  "train_image_data_1.parquet.zip",
                  "train_image_data_2.parquet.zip",
                  "train_image_data_3.parquet.zip"]
SIZE = 128
WIDTH = 236
HEIGHT = 137


if __name__ == "__main__":
    print("INPUT_DIR: ", INPUT_DIR)
    train_df = pd.read_csv(INPUT_DIR / "train.csv")
    for zip_file in TRAIN_ZIPFILES:
        dir_name = zip_file.split(".")[0]
        Path(TRAIN_DIR / dir_name).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(INPUT_DIR / zip_file) as existing_zip:
            parquet_file = TRAIN_DIR / "{}.parquet".format(dir_name)
            if os.path.exists(parquet_file) is False:
                existing_zip.extractall(TRAIN_DIR)
            img_df = pd.read_parquet(parquet_file)
            for idx in tqdm(range(len(img_df))):
                img0 = 255 - img_df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
                img = (img0 * (255.0 / img0.max())).astype(np.uint8)
                img = crop_and_resize_img(img, SIZE, HEIGHT, WIDTH)
                name = img_df.ilock[idx, 0]
                cv2.imwrite(str(TRAIN_DIR / f"{name}.png"), img)
                assert False
