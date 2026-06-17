import ee

ee.Initialize(project="bharatiya-hackathon")

collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

print("Images found:", collection.size().getInfo())
print("Earth Engine Connected Successfully!")