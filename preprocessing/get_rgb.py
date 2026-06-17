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

rgb = geemap.ee_to_numpy(
    image.select(["SR_B4", "SR_B3", "SR_B2"]),
    region=roi
)

print("RGB Shape:")
print(rgb.shape)