import ee
import geemap

ee.Initialize(project="bharatiya-hackathon")

roi = ee.Geometry.Rectangle(
    [76.8, 28.4, 77.4, 28.9]
)

collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

print("Images found:", collection.size().getInfo())

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

    rgb = get_rgb(image)
    tir = get_tir(image)

    print(rgb.shape)
    print(tir.shape)

if __name__ == "__main__":
    main()