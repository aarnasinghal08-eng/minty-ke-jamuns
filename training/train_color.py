import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Adjust paths to import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.color_model import ColorizationUNet, PatchGANDiscriminator
from models.losses import heteroscedastic_loss
from training.dataset import ColorizationDataset

def train_color(epochs=5, batch_size=8, lr=2e-4, lambda_l1=100.0, lambda_gan=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Dataset
    dataset = ColorizationDataset(processed_dir="dataset/processed")
    
    # 2. Train/Validation Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize Models
    generator = ColorizationUNet(in_channels=1).to(device)
    discriminator = PatchGANDiscriminator(in_channels=4).to(device) # TIR (1ch) + RGB (3ch) = 4ch
    
    # 4. Initialize Optimizers
    optimizer_G = optim.AdamW(generator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    
    # 5. Define loss metrics
    l1_criterion = nn.L1Loss()
    gan_criterion = nn.BCEWithLogitsLoss()
    
    # Create checkpoints directory in outputs
    os.makedirs("outputs/checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    
    print("\nStarting Pix2Pix Colorization (TIR -> RGB) Adversarial Training...")
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        train_loss_G = 0.0
        train_loss_D = 0.0
        
        for thermal, rgb_target in train_loader:
            thermal = thermal.to(device)
            rgb_target = rgb_target.to(device)
            
            # ----------------------------------------
            # Step 1: Train Discriminator (Critic)
            # ----------------------------------------
            optimizer_D.zero_grad()
            
            # Loss on Real image pair
            pred_real = discriminator(thermal, rgb_target)
            loss_D_real = gan_criterion(pred_real, torch.ones_like(pred_real))
            
            # Loss on Fake generated image pair
            pred_rgb, pred_var = generator(thermal)
            pred_fake = discriminator(thermal, pred_rgb.detach())
            loss_D_fake = gan_criterion(pred_fake, torch.zeros_like(pred_fake))
            
            # Combined Discriminator updates
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()
            
            train_loss_D += loss_D.item() * thermal.size(0)
            
            # ----------------------------------------
            # Step 2: Train Generator (Artist)
            # ----------------------------------------
            optimizer_G.zero_grad()
            
            # GAN adversarial loss: make discriminator believe the generated image is real
            pred_fake_for_G = discriminator(thermal, pred_rgb)
            loss_G_gan = gan_criterion(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            
            # Direct L1 Pixel mapping loss
            loss_G_l1 = l1_criterion(pred_rgb, rgb_target)
            
            # Physics-aware uncertainty weighting loss
            loss_G_unc = heteroscedastic_loss(pred_rgb, rgb_target, pred_var)
            
            # Combined Generator updates
            loss_G = lambda_gan * loss_G_gan + lambda_l1 * loss_G_l1 + loss_G_unc
            loss_G.backward()
            optimizer_G.step()
            
            train_loss_G += loss_G.item() * thermal.size(0)
            
        train_loss_G /= len(train_dataset)
        train_loss_D /= len(train_dataset)
        
        # 6. Validation Loop
        generator.eval()
        val_loss = 0.0
        val_l1 = 0.0
        with torch.no_grad():
            for thermal, rgb_target in val_loader:
                thermal = thermal.to(device)
                rgb_target = rgb_target.to(device)
                
                pred_rgb, pred_var = generator(thermal)
                loss_unc = heteroscedastic_loss(pred_rgb, rgb_target, pred_var)
                loss_l1 = l1_criterion(pred_rgb, rgb_target)
                
                total_loss = loss_unc + lambda_l1 * loss_l1
                val_loss += total_loss.item() * thermal.size(0)
                val_l1 += loss_l1.item() * thermal.size(0)
                
        val_loss /= len(val_dataset)
        val_l1 /= len(val_dataset)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss G: {train_loss_G:.4f} | Loss D: {train_loss_D:.4f} | Val Loss: {val_loss:.4f} | Val L1: {val_l1:.4f}")
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), "outputs/checkpoints/colorization_best.pth")
            print("Saved new best generator checkpoint!")
            
    print("Colorization Training Completed.")

if __name__ == "__main__":
    # Dry run 2 epochs to verify execution
    train_color(epochs=2, batch_size=4, lr=2e-4)
