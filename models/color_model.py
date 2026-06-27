import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                   diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ColorizationUNet(nn.Module):
    """
    Generator Model: U-Net mapping single-channel thermal input
    to RGB + Uncertainty (Variance).
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        
        # RGB Output Head (Sigmoid normalized to [0, 1])
        self.rgb_head = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )
        # Uncertainty Variance Output Head (Softplus to keep strictly positive)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softplus()
        )

    def forward(self, x):
        # input shape: B, 1, H, W (e.g. 128x128 TIR)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        rgb = self.rgb_head(x)
        variance = self.uncertainty_head(x) + 1e-6 # epsilon for numerical stability
        
        return rgb, variance

class PatchGANDiscriminator(nn.Module):
    """
    Discriminator Model: 70x70 PatchGAN.
    Takes concatenated Condition (TIR) + Image (RGB) and predicts real/fake logit maps.
    """
    def __init__(self, in_channels=4): # 1-ch TIR + 3-ch RGB = 4 channels
        super().__init__()
        
        def block(in_c, out_c, stride=2, use_bn=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
            
        self.model = nn.Sequential(
            block(in_channels, 64, stride=2, use_bn=False),
            block(64, 128, stride=2, use_bn=True),
            block(128, 256, stride=2, use_bn=True),
            block(256, 512, stride=1, use_bn=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, condition, image):
        x = torch.cat([condition, image], dim=1)
        return self.model(x)

if __name__ == "__main__":
    generator = ColorizationUNet(in_channels=1)
    discriminator = PatchGANDiscriminator(in_channels=4)
    
    dummy_tir = torch.randn(2, 1, 128, 128)
    dummy_rgb = torch.randn(2, 3, 128, 128)
    
    pred_rgb, pred_var = generator(dummy_tir)
    pred_d = discriminator(dummy_tir, dummy_rgb)
    
    print("Generator Output RGB shape:", pred_rgb.shape)
    print("Generator Output Variance shape:", pred_var.shape)
    print("Discriminator Output shape:", pred_d.shape)
