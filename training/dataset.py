import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class ColorizationDataset(Dataset):
    """
    Dataset class to load TIR input patches and corresponding RGB target patches
    directly from the unified .npz files created by process_patches.py.
    """
    def __init__(self, processed_dir="dataset/processed"):
        self.processed_dir = processed_dir
        self.patch_files = sorted(glob.glob(os.path.join(processed_dir, "patch_*.npz")))
        
        self.use_mock = len(self.patch_files) == 0
        if self.use_mock:
            print(f"WARNING: No patch files found in '{processed_dir}'. Using mock fallback dataset.")
            self.mock_size = 50
        else:
            print(f"Loaded {len(self.patch_files)} patches from '{processed_dir}'.")

    def __len__(self):
        return self.mock_size if self.use_mock else len(self.patch_files)

    def __getitem__(self, idx):
        if self.use_mock:
            # Generate random mock data of correct shape
            tir = np.random.rand(1, 128, 128).astype(np.float32)
            rgb = np.random.rand(3, 128, 128).astype(np.float32)
        else:
            # Load the unified .npz file
            data = np.load(self.patch_files[idx])
            
            # Extract high-resolution 100m TIR image (thermal input)
            tir_raw = data["tir_100m"]
            if tir_raw.ndim == 2:
                tir = np.expand_dims(tir_raw, axis=0) # Shape: 1, 128, 128
            else:
                tir = tir_raw
                
            # Extract high-resolution RGB image (visible target)
            rgb_raw = data["rgb"]
            if rgb_raw.ndim == 3 and rgb_raw.shape[0] != 3:
                # Transpose H, W, C -> C, H, W (3, 128, 128)
                rgb = np.transpose(rgb_raw, (2, 0, 1))
            else:
                rgb = rgb_raw

        return torch.tensor(tir, dtype=torch.float32), torch.tensor(rgb, dtype=torch.float32)
