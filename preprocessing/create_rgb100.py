import os
import cv2
import numpy as np

SCALE_100 = 30 / 100

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

rgb_folder = "dataset/rgb"
files = sorted(os.listdir(rgb_folder))
for file in files:

    rgb = np.load(os.path.join(rgb_folder, file))
    rgb100 = downsample(rgb,SCALE_100)

    np.save(f"dataset/rgb100/{file}",rgb100)