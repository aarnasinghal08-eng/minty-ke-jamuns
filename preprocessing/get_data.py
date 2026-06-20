import ee
import geemap
import numpy as np
import matplotlib.pyplot as plt

ee.Initialize(project="bharatiya-hackathon")

roi = ee.Geometry.Rectangle(
    [77.00, 28.55, 77.15, 28.70]
)

collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

#print("Images found:", collection.size().getInfo())

def get_best_landsat_image():
    image = (
    collection
    .filterBounds(roi)
    .filterDate("2024-01-01", "2024-12-31")
    .sort("CLOUD_COVER")
    .first()
    )
    return image

def get_rgb(image):
    rgb = geemap.ee_to_numpy(
    image.select(["SR_B4", "SR_B3", "SR_B2"]),
    region=roi
    )
    return rgb

def get_tir(image):
    tir = geemap.ee_to_numpy(
    image.select(["ST_B10"]),
    region=roi
    )
    return tir

def main():
    image = get_best_landsat_image()
#    print(image.geometry().bounds().getInfo())

    rgb = get_rgb(image)
    tir = get_tir(image)

#    print(rgb.shape)
#    print(tir.shape)

#    print(np.count_nonzero(rgb))
#    print(np.count_nonzero(tir))

#    mask = np.any(rgb > 0, axis=2)

#    print(mask.sum())

    np.save("dataset/rgb/rgb-001.npy",rgb)
    np.save("dataset/tir/tir-001.npy",tir)

    loaded_rgb = np.load("dataset/rgb/rgb-001.npy")
    
    rgb_normalized = (
    loaded_rgb - loaded_rgb.min()
    ) / (
    loaded_rgb.max() - loaded_rgb.min()
    )
    plt.imshow(rgb_normalized)
    plt.show()

if __name__ == "__main__":
    main()