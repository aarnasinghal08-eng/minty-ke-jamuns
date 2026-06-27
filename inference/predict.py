import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image

# Adjust paths to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.color_model import ColorizationUNet

def scale_temp(band):
    # Scale Landsat-9 Surface Temperature: ST_K = DN * 0.00341802 + 149.0
    st_kelvin = band * 0.00341802 + 149.0
    # Convert to Celsius
    st_celsius = st_kelvin - 273.15
    # Normalize to [0, 1] assuming range of -20C to 60C
    normalized = (st_celsius - (-20.0)) / (60.0 - (-20.0))
    return np.clip(normalized, 0.0, 1.0)

def generate_confidence_overlay(variance_map):
    # Normalize variance to [0, 1]
    v_min = variance_map.min()
    v_max = variance_map.max()
    if v_max - v_min > 1e-8:
        norm_var = (variance_map - v_min) / (v_max - v_min)
    else:
        norm_var = np.zeros_like(variance_map)
        
    confidence = 1.0 - norm_var
    h, w = confidence.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Map confidence to Green (High) -> Yellow -> Red (Low)
    for y in range(h):
        for x in range(w):
            conf = confidence[y, x]
            if conf > 0.7:
                t = (conf - 0.7) / 0.3
                overlay[y, x] = [int(220 * (1 - t)), int(220 * (1 - t) + 200 * t), 0]
            elif conf > 0.4:
                t = (conf - 0.4) / 0.3
                overlay[y, x] = [int(200 * (1 - t) + 220 * t), int(220 * t), 0]
            else:
                overlay[y, x] = [200, 0, 0]
                
    return overlay

def predict(region_name="thar"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")
    
    # 1. Paths
    tir_path = f"dataset/raw/{region_name}_tir.npy"
    weights_path = "outputs/checkpoints/colorization_best.pth"
    output_dir = "outputs/predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(tir_path):
        print(f"Error: Raw thermal file '{tir_path}' not found! Run preprocessing/get_data.py first.")
        return
        
    # 2. Load and scale TIR image
    tir_raw = np.load(tir_path)
    # get_data.py saves TIR as (H, W, 1) — squeeze the trailing channel dim to get (H, W)
    if tir_raw.ndim == 3 and tir_raw.shape[-1] == 1:
        tir_raw = np.squeeze(tir_raw, axis=-1)  # (H, W, 1) -> (H, W)
    elif tir_raw.ndim == 3 and tir_raw.shape[0] == 1:
        tir_raw = np.squeeze(tir_raw, axis=0)   # (1, H, W) -> (H, W)

    tir_scaled = scale_temp(tir_raw)
    print(f"TIR loaded: shape={tir_raw.shape}, scaled min={tir_scaled.min():.3f}, max={tir_scaled.max():.3f}")
    
    # Slice a 128x128 center patch for the model
    h, w = tir_scaled.shape
    start_h = (h - 128) // 2
    start_w = (w - 128) // 2
    tir_patch = tir_scaled[start_h:start_h+128, start_w:start_w+128]
    print(f"Cropped patch shape: {tir_patch.shape}")
    
    # Format to tensor (B=1, C=1, H=128, W=128)
    tir_tensor = torch.tensor(tir_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Load Model
    generator = ColorizationUNet(in_channels=1).to(device)
    if os.path.exists(weights_path):
        generator.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded trained model weights successfully.")
    else:
        print("WARNING: 'colorization_best.pth' not found. Using randomly initialized weights.")
        
    generator.float() # Force all weights to Float32 precision
    generator.eval()
    
    # Diagnostic prints
    print(f"tir_tensor shape: {tir_tensor.shape}, dtype: {tir_tensor.dtype}, device: {tir_tensor.device}")
    for name, param in generator.named_parameters():
        print(f"Model param '{name}' dtype: {param.dtype}, device: {param.device}")
        break
        
    # 4. Forward pass
    try:
        with torch.no_grad():
            pred_rgb_tensor, pred_var_tensor = generator(tir_tensor)
    except Exception as e:
        print("\n=== CRITICAL FORWARD PASS EXCEPTION ===")
        print(type(e), str(e))
        import traceback
        traceback.print_exc()
        raise e
        
    # Convert prediction back to numpy range [0, 255]
    with torch.no_grad():
        pred_rgb = pred_rgb_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        pred_rgb = (np.clip(pred_rgb, 0.0, 1.0) * 255).astype(np.uint8)
        
        pred_var = pred_var_tensor.squeeze().cpu().numpy()
        
    # Generate confidence mask
    confidence_overlay = generate_confidence_overlay(pred_var)
    
    # Save input TIR, reconstructed RGB, and confidence map
    Image.fromarray((tir_patch * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{region_name}_input_tir.png"))
    Image.fromarray(pred_rgb).save(os.path.join(output_dir, f"{region_name}_output_rgb.png"))
    Image.fromarray(confidence_overlay).save(os.path.join(output_dir, f"{region_name}_confidence.png"))
    
    print(f"\nInference completed successfully!")
    print(f"Outputs saved to folder: '{output_dir}/'")
    print(f"1. Input Thermal: {region_name}_input_tir.png")
    print(f"2. Colorized RGB: {region_name}_output_rgb.png")
    print(f"3. Confidence Map: {region_name}_confidence.png")

if __name__ == "__main__":
    # Run prediction on Delhi (urban) or Thar (desert)
    predict("delhi")
