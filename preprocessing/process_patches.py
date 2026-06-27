import os
import glob
import numpy as np
import cv2

TERRAINS = {
    "thar": 0,
    "delhi": 1,
    "kutch": 2,
    "ladakh": 3,
    "telangana": 4
}

def scale_temp(band):
    st_kelvin = band * 0.00341802 + 149.0
    st_celsius = st_kelvin - 273.15
    normalized = (st_celsius - (-20.0)) / (60.0 - (-20.0))
    return np.clip(normalized, 0.0, 1.0)

def main(stride=4):
    raw_dir = "dataset/raw"
    processed_dir = "dataset/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Remove any old patches
    for f in glob.glob(os.path.join(processed_dir, "patch_*.npz")):
        os.remove(f)

    patch_count = 0
    print("Generating patches from raw Landsat data...")

    for region, terrain_idx in TERRAINS.items():
        rgb_path = os.path.join(raw_dir, f"{region}_rgb.npy")
        tir_path = os.path.join(raw_dir, f"{region}_tir.npy")
        nir_path = os.path.join(raw_dir, f"{region}_nir.npy")

        if not (os.path.exists(rgb_path) and os.path.exists(tir_path) and os.path.exists(nir_path)):
            print(f"  Skipping '{region}': missing raw files.")
            continue

        print(f"  Processing region: {region}...")

        rgb_raw = np.load(rgb_path).astype(np.float32)
        tir_raw = np.load(tir_path).astype(np.float32)
        nir_raw = np.load(nir_path).astype(np.float32)

        # Squeeze trailing channel dim: (H, W, 1) -> (H, W)
        if tir_raw.ndim == 3 and tir_raw.shape[-1] == 1:
            tir_raw = np.squeeze(tir_raw, axis=-1)
        if nir_raw.ndim == 3 and nir_raw.shape[-1] == 1:
            nir_raw = np.squeeze(nir_raw, axis=-1)

        # Scale TIR to [0,1]
        tir_scaled = scale_temp(tir_raw)

        # Scale RGB and NIR using Landsat-9 SR formula
        rgb_scaled = np.clip(rgb_raw * 0.0000275 - 0.2, 0.0, 1.0)
        nir_scaled = np.clip(nir_raw * 0.0000275 - 0.2, 0.0, 1.0)

        # NDVI = (NIR - Red) / (NIR + Red)  [Red = channel 0 = SR_B4]
        red = rgb_scaled[..., 0]
        denom = nir_scaled + red
        ndvi = np.where(denom > 1e-8, (nir_scaled - red) / denom, 0.0)
        ndvi = np.clip(ndvi, -1.0, 1.0)

        h, w = tir_scaled.shape
        patch_size = 128
        region_patches = 0

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                rgb_patch    = rgb_scaled[y:y+patch_size, x:x+patch_size]          # (128,128,3)
                ndvi_patch   = ndvi[y:y+patch_size, x:x+patch_size]                # (128,128)
                tir_100m     = tir_scaled[y:y+patch_size, x:x+patch_size]          # (128,128)
                tir_200m     = cv2.resize(tir_100m, (64, 64), interpolation=cv2.INTER_AREA)  # (64,64)

                np.savez_compressed(
                    os.path.join(processed_dir, f"patch_{patch_count:06d}.npz"),
                    rgb=rgb_patch,
                    ndvi=ndvi_patch,
                    tir_100m=tir_100m,
                    tir_200m=tir_200m,
                    terrain_idx=terrain_idx
                )
                patch_count += 1
                region_patches += 1

        print(f"    -> {region_patches} patches created.")

    print(f"\nDone! Total patches: {patch_count}  saved to '{processed_dir}/'")

if __name__ == "__main__":
    main(stride=4)
