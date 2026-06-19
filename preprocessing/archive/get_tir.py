import ee
import geemap

ee.Initialize(project="bharatiya-hackathon")

roi = ee.Geometry.Rectangle(
    [76.8, 28.4, 77.4, 28.9]
)

image = (
    ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    .filterBounds(roi)
    .filterDate("2024-01-01", "2024-12-31")
    .sort("CLOUD_COVER")
    .first()
)

tir = geemap.ee_to_numpy(
    image.select(["ST_B10"]),
    region=roi
)

print("TIR Shape: ")
print(tir.shape)