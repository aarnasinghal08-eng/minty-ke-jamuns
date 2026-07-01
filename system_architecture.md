# 🛰️ System Architecture — Simplified Pipeline Flow

Below is the end-to-end pipeline showing how raw satellite thermal data is transformed into enhanced, high-resolution visible color images.

---

## E2E Pipeline Diagram

```mermaid
flowchart TD
    subgraph STAGE_1["🛰️ 1. DATA INGESTION & PREPARATION"]
        A1["Download Landsat-9 Satellite Imagery\n(TIR, RGB, & NIR Bands)"]
        A2["Calibrate Raw Thermal Values\n(Convert to Celsius Scale & Normalize)"]
        A3["Create Paired Image Patches\n(128×128 pixel resolution)"]
        A1 --> A2 --> A3
    end

    subgraph STAGE_2["🔬 2. SPATIAL SUPER-RESOLUTION"]
        B1["Input: Blurry Low-Res Thermal Image\n(200-meter resolution)"]
        B2["SwinIR Transformer Model\n(Extracts local details and textures)"]
        B3["Output: Sharp High-Res Thermal Image\n(100-meter resolution)"]
        B1 --> B2 --> B3
    end

    subgraph STAGE_3["🎨 3. SPECTRAL COLORIZATION"]
        C1["U-Net Generator Model\n(Translates thermal spectrum to visible RGB colors)"]
        C2["Uncertainty Prediction Head\n(Locates areas with low prediction confidence)"]
        C3["PatchGAN Discriminator Model\n(Critiques generated colors to ensure realism)"]
        
        B3 --> C1
        C1 --> C2
        C1 -.->|"Training Feedback"| C3
    end

    subgraph STAGE_4["📉 4. MULTI-OBJECTIVE OPTIMIZATION"]
        D1["Adversarial Loss\n(Ensures natural color output)"]
        D2["Pixel L1 Loss\n(Ensures color intensity accuracy)"]
        D3["Heteroscedastic Loss\n(Balances metrics using uncertainty)"]
        D4["Combined Network Loss\n(Updates model weights)"]
        
        D1 & D2 & D3 --> D4
    end

    subgraph STAGE_5["🖥️ 5. INTERACTIVE DASHBOARD"]
        E1["Load Trained Model Weights\n(colorization_best.pth)"]
        E2["Generate Output Images\n• Synthesized Color Image\n• Prediction Confidence Overlay"]
        E3["Quality Metrics Assessment\n• With Reference: PSNR & SSIM\n• Without Reference: Entropy & Sharpness"]
        
        C1 --> E1
        E1 --> E2 --> E3
    end
```

---

## Quick Component Reference

| Stage | Input | Target | Purpose |
|---|---|---|---|
| **Data Prep** | Raw Satellite Data | Normalized Patches | Cleans and structures data for model training |
| **Super-Resolution** | Blurry Thermal | Sharp Thermal | Doubles image detail and sharpens terrain structures |
| **Colorization** | Sharp Thermal | Colorized RGB | Synthesizes realistic visible bands from heat maps |
| **Dashboard** | Thermal Image Upload | Enhanced Visuals + Metrics | Interactive real-time testing and performance scoring |
