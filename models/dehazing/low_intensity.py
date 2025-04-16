import torch
import torch.nn as nn
from .base_model import BaseDehazeModel, ConvBlock, ResidualBlock

class LightweightDehazeModel(BaseDehazeModel):
    """Lightweight dehazing model for low-intensity fog"""
    
    def __init__(self, in_channels=3, base_channels=32, n_blocks=3):
        super(LightweightDehazeModel, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        
        # Initial feature extraction
        self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Main processing blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(n_blocks)]
        )
        
        # Final reconstruction
        self.output_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels, kernel_size=3, padding=1),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Skip connection handling
        self.skip_alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # Extract features
        features = self.init_conv(x)
        
        # Process through residual blocks
        features = self.residual_blocks(features)
        
        # Generate output image
        out = self.output_conv(features)
        
        # Apply weighted skip connection
        # For light haze, we want to preserve a lot of the original image
        return (1 - self.skip_alpha) * x + self.skip_alpha * out
    
    def get_info(self):
        info = super().get_info()
        info.update({
            "model_type": "LightweightDehazeModel",
            "base_channels": self.base_channels,
            "n_blocks": self.n_blocks
        })
        return info

class LowIntensityDehazeModel(BaseDehazeModel):
    """Enhanced dehazing model for low-intensity fog"""
    
    def __init__(self, in_channels=3, base_channels=32, n_blocks=3):
        super(LowIntensityDehazeModel, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        
        # Initial feature extraction
        self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.down1 = nn.Sequential(
            ConvBlock(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            ResidualBlock(base_channels*2)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(base_channels*2) for _ in range(n_blocks-1)]
        )
        
        # Decoder (upsampling path)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final reconstruction
        self.output_conv = nn.Sequential(
            ConvBlock(base_channels*2, base_channels, kernel_size=3, padding=1),  # Concat with skip
            ConvBlock(base_channels, base_channels, kernel_size=3, padding=1),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Initial features
        init_features = self.init_conv(x)
        
        # Encoder
        down_features = self.down1(init_features)
        
        # Bottleneck
        bottle_features = self.bottleneck(down_features)
        
        # Decoder
        up_features = self.up1(bottle_features)
        
        # Skip connection - concatenate with initial features
        concat_features = torch.cat([up_features, init_features], dim=1)
        
        # Final output
        out = self.output_conv(concat_features)
        
        # Apply residual connection for low-intensity haze
        # We assume the network is estimating the residual (clean - hazy)
        return torch.clamp(x + (out - 0.5) * 2, 0, 1)  # Scale residual to [-1, 1]
    
    def get_info(self):
        info = super().get_info()
        info.update({
            "model_type": "LowIntensityDehazeModel",
            "base_channels": self.base_channels,
            "n_blocks": self.n_blocks
        })
        return info

def create_low_intensity_model(config):
    """Create dehazing model for low intensity fog based on config"""
    model_type = config['dehazing']['low']['model_type']
    
    if model_type == 'lightweight':
        return LightweightDehazeModel(
            base_channels=config['dehazing']['low']['channels'],
            n_blocks=config['dehazing']['low']['blocks']
        )
    else:
        return LowIntensityDehazeModel(
            base_channels=config['dehazing']['low']['channels'],
            n_blocks=config['dehazing']['low']['blocks']
        )