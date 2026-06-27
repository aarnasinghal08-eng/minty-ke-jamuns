import ee
import geemap
import numpy as np
import os

# Initialize Earth Engine
def init_ee():
    try:
        ee.Initialize(project="bharatiya-hackathon")
        print("Earth Engine Initialized with project 'bharatiya-hackathon'!")
    except Exception as e:
        print("Project-based initialization failed. Trying default Initialize...")
        try:
            ee.Initialize()
            print("Earth Engine Initialized successfully!")
        except Exception as ex:
            print("Failed to initialize Earth Engine. Please authenticate by running 'earthengine authenticate' in the terminal.")
            raise ex

# Define the 5 regions of interest with their labels and coordinates
# Approximately 0.15 x 0.15 degrees (~15km x 15km) to stay within GEE memory limits
REGIONS = {
    "thar": {
        "label": "desert",
        "roi": [71.50, 26.50, 71.65, 26.65] # Desert (Rajasthan)
    },
    "delhi": {
        "label": "urban",
        "roi": [77.00, 28.55, 77.15, 28.70] # Urban (Capital)
    },
    "kutch": {
        "label": "coastal",
        "roi": [69.80, 22.80, 69.95, 22.95] # Coastal / Salt Flat (Gujarat)
    },
    "ladakh": {
        "label": "mountain",
        "roi": [77.50, 34.10, 77.65, 34.25] # Mountain (Cold Desert)
    },
    "telangana": {
        "label": "agriculture",
        "roi": [78.50, 17.50, 78.65, 17.65] # Agriculture (Deccan)
    }
}

def get_best_image(roi_geometry):
    collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    image = (
        collection
        .filterBounds(roi_geometry)
        .filterDate("2024-01-01", "2024-12-31")
        .sort("CLOUD_COVER")
        .first()
    )
    return image

def main():
    init_ee()
    
    # Create dataset directories
    os.makedirs("dataset/raw", exist_ok=True)
    
    for name, info in REGIONS.items():
        print(f"\nProcessing region: {name} (Terrain: {info['label']})...")
        coords = info["roi"]
        roi = ee.Geometry.Rectangle(coords)
        
        image = get_best_image(roi)
        if image is None:
            print(f"No image found for region {name}!")
            continue
            
        print(f"Selected image ID: {image.id().getInfo()}")
        
        # Download RGB bands (SR_B4, SR_B3, SR_B2) resampled to 100m scale
        print("Downloading RGB bands at 100m scale...")
        rgb_ee = image.select(["SR_B4", "SR_B3", "SR_B2"])
        rgb_np = geemap.ee_to_numpy(rgb_ee, region=roi, scale=100)
        
        # Download NIR band (SR_B5) resampled to 100m scale
        print("Downloading NIR band at 100m scale...")
        nir_ee = image.select(["SR_B5"])
        nir_np = geemap.ee_to_numpy(nir_ee, region=roi, scale=100)
        
        # Download Thermal band (ST_B10) at 100m scale (native)
        print("Downloading TIR band (ST_B10) at 100m scale...")
        tir_ee = image.select(["ST_B10"])
        tir_np = geemap.ee_to_numpy(tir_ee, region=roi, scale=100)
        
        # Check download sanity
        print(f"RGB Shape: {rgb_np.shape}, Non-zero count: {np.count_nonzero(rgb_np)}")
        print(f"NIR Shape: {nir_np.shape}, Non-zero count: {np.count_nonzero(nir_np)}")
        print(f"TIR Shape: {tir_np.shape}, Non-zero count: {np.count_nonzero(tir_np)}")
        
        # Save to disk
        np.save(f"dataset/raw/{name}_rgb.npy", rgb_np)
        np.save(f"dataset/raw/{name}_nir.npy", nir_np)
        np.save(f"dataset/raw/{name}_tir.npy", tir_np)
        print(f"Successfully saved raw arrays for {name}.")

if __name__ == "__main__":
    main()