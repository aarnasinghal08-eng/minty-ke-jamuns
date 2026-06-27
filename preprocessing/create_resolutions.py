import os
import cv2
import numpy as np

SCALE_100 = 30 / 100
SCALE_200 = 30 / 200


def downsample(image, scale):

    height = image.shape[0]
    width = image.shape[1]

    new_height = int(height * scale)
    new_width = int(width * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized

tir_folder = "dataset/tir"
files = sorted(os.listdir(tir_folder))
for file in files:

    tir = np.load(os.path.join(tir_folder, file))
    tir100 = downsample(tir,SCALE_100)
    tir200 = downsample(tir,SCALE_200)

    np.save(f"dataset/tir100/{file}",tir100)
    np.save(f"dataset/tir200/{file}",tir200)

    print(file,tir.shape,tir100.shape,tir200.shape)