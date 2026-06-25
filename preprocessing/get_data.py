import ee
import geemap
import numpy as np
import matplotlib.pyplot as plt

ee.Initialize(project="bharatiya-hackathon")

ROIS = [
    ("Delhi", [77.00, 28.55, 77.15, 28.70]),
    ("Mumbai", [72.80, 18.90, 73.05, 19.20]),
    ("Kolkata", [88.25, 22.45, 88.50, 22.70]),
    ("Bangalore", [77.45, 12.85, 77.75, 13.15]),
    ("Jaipur", [75.70, 26.80, 76.00, 27.10]),
]

collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

def get_best_landsat_image(roi):
    image = (
    collection
    .filterBounds(roi)
    .filterDate("2024-01-01", "2024-12-31")
    .sort("CLOUD_COVER")
    .first()
    )
    return image

def get_rgb(image,roi):
    rgb = geemap.ee_to_numpy(
    image.select(["SR_B4", "SR_B3", "SR_B2"]),
    region=roi
    )
    return rgb

def get_tir(image,roi):
    tir = geemap.ee_to_numpy(
    image.select(["ST_B10"]),
    region=roi
    )
    return tir

def main():
    for idx, (city_name, coords) in enumerate(ROIS):
        roi = ee.Geometry.Rectangle(coords)
        image = get_best_landsat_image(roi)
        rgb = get_rgb(image,roi)
        tir = get_tir(image,roi)
        np.save(f"dataset/rgb/scene_{idx:03d}_rgb.npy",rgb)
        np.save(f"dataset/tir/scene_{idx:03d}_tir.npy",tir)

    print("Dataset Generation Complete!")

if __name__ == "__main__":
    main()