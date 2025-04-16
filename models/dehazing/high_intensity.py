import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseDehazeModel, ConvBlock, ResidualBlock, AttentionBlock, EncoderDecoder

class HighIntensityDehazeModel(BaseDehazeModel):
    """Complex dehazing model for high-intensity fog with attention mechanisms"""
    
    def __init__(self, in_channels=3, base_channels=96, n_blocks=9):
        super(HighIntensityDehazeModel, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        
        # Initial feature extraction
        self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        channels = [base_channels, base_channels*2, base_channels*4]
        
        # First downsampling block with attention
        self.encoder.append(
            nn.Sequential(
                ConvBlock(channels[0], channels[1], kernel_size=4, stride=2, padding=1),
                ResidualBlock(channels[1]),
                ResidualBlock(channels[1]),
                AttentionBlock(channels[1])
            )
        )
        
        # Second downsampling block with attention
        self.encoder.append(
            nn.Sequential(
                ConvBlock(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
                ResidualBlock(channels[2]),
                ResidualBlock(channels[2]),
                AttentionBlock(channels[2])
            )
        )
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[2]),
            AttentionBlock(channels[2]),
            ResidualBlock(channels[2]),
            AttentionBlock(channels[2])
        )
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        
        # First upsampling block
        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(inplace=True),
                ResidualBlock(channels[1]),
                AttentionBlock(channels[1])
            )
        )
        
        # Second upsampling block
        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(channels[1]*2, channels[0], kernel_size=4, stride=2, padding=1),  # *2 for skip connection
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True),
                ResidualBlock(channels[0]),
                AttentionBlock(channels[0])
            )
        )
        
        # Final reconstruction
        self.output_conv = nn.Sequential(
            ConvBlock(channels[0]*2, channels[0], kernel_size=3, padding=1),  # *2 for skip connection
            ConvBlock(channels[0], channels[0]//2, kernel_size=3, padding=1),
            nn.Conv2d(channels[0]//2, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Additional guided filter for detail preservation
        self.detail_branch = nn.Sequential(
            ConvBlock(in_channels, 16, kernel_size=3, padding=1),
            ConvBlock(16, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract guidance map for detail preservation
        guidance = self.detail_branch(x)
        
        # Initial features
        features = [self.init_conv(x)]  # Store for skip connections
        
        # Encoder
        for enc_block in self.encoder:
            features.append(enc_block(features[-1]))
        
        # Bottleneck
        bottle_features = self.bottleneck(features[-1])
        
        # Decoder with skip connections
        x1 = self.decoder[0](bottle_features)
        
        # Ensure dimensions match before concatenation
        if x1.shape[2:] != features[-2].shape[2:]:
            x1 = nn.functional.interpolate(
                x1, 
                size=features[-2].shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        x1 = torch.cat([x1, features[-2]], dim=1)  # Skip connection
        
        x2 = self.decoder[1](x1)
        
        # Ensure dimensions match before concatenation
        if x2.shape[2:] != features[0].shape[2:]:
            x2 = nn.functional.interpolate(
                x2, 
                size=features[0].shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        x2 = torch.cat([x2, features[0]], dim=1)  # Skip connection
        
        # Final output - predict residual
        residual = self.output_conv(x2)
        
        # Scale residual based on guidance (stronger modifications where fog is heavier)
        weighted_residual = residual * guidance
        
        # Apply to input
        return torch.clamp(x + weighted_residual, 0, 1)
    
    def get_info(self):
        info = super().get_info()
        info.update({
            "model_type": "HighIntensityDehazeModel",
            "base_channels": self.base_channels,
            "n_blocks": self.n_blocks
        })
        return info

class DualBranchAttentionModel(BaseDehazeModel):
    """Complex model with dual branch processing for high-intensity fog"""
    
    def __init__(self, in_channels=3, base_channels=96, n_blocks=9):
        super(DualBranchAttentionModel, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        
        # Global branch for overall structure
        self.global_branch = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(base_channels),
            AttentionBlock(base_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(base_channels),
            AttentionBlock(base_channels),
            ResidualBlock(base_channels),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ResidualBlock(base_channels),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(base_channels, base_channels//2, kernel_size=3, padding=1)
        )
        
        # Local branch for detail preservation
        self.local_branch = nn.Sequential(
            ConvBlock(in_channels, base_channels//2, kernel_size=3, padding=1),
            ResidualBlock(base_channels//2),
            ResidualBlock(base_channels//2),
            ConvBlock(base_channels//2, base_channels//2, kernel_size=3, padding=1)
        )
        
        # Transmission map estimation
        self.transmission_branch = nn.Sequential(
            ConvBlock(base_channels, base_channels//2, kernel_size=3, padding=1),
            ConvBlock(base_channels//2, base_channels//4, kernel_size=3, padding=1),
            nn.Conv2d(base_channels//4, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.fusion_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels//2, kernel_size=3, padding=1),
            nn.Conv2d(base_channels//2, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Process through branches
        global_features = self.global_branch(x)
        local_features = self.local_branch(x)
        
        # Concatenate features
        concat_features = torch.cat([global_features, local_features], dim=1)
        
        # Estimate transmission map (indicates fog density)
        transmission = self.transmission_branch(concat_features)
        
        # Generate final output
        residual = self.fusion_conv(concat_features)
        
        # Apply residual scaled by transmission map
        # Lower transmission (denser fog) means stronger adjustments
        return torch.clamp(x + (1 - transmission) * residual, 0, 1)
    
    def get_info(self):
        info = super().get_info()
        info.update({
            "model_type": "DualBranchAttentionModel",
            "base_channels": self.base_channels,
            "n_blocks": self.n_blocks
        })
        return info

def create_high_intensity_model(config):
    """Create dehazing model for high intensity fog based on config"""
    model_type = config['dehazing']['high']['model_type']
    
    if model_type == 'dual_branch':
        return DualBranchAttentionModel(
            base_channels=config['dehazing']['high']['channels'],
            n_blocks=config['dehazing']['high']['blocks']
        )
    else:
        # Always use the stable HighIntensityDehazeModel instead of EncoderDecoder
        return HighIntensityDehazeModel(
            base_channels=config['dehazing']['high']['channels'],
            n_blocks=config['dehazing']['high']['blocks']
        )
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .base_model import BaseDehazeModel, ConvBlock, ResidualBlock, AttentionBlock, EncoderDecoder

# class HighIntensityDehazeModel(BaseDehazeModel):
#     """Complex dehazing model for high-intensity fog with attention mechanisms"""
    
#     def __init__(self, in_channels=3, base_channels=96, n_blocks=9):
#         super(HighIntensityDehazeModel, self).__init__()
        
#         self.in_channels = in_channels
#         self.base_channels = base_channels
#         self.n_blocks = n_blocks
        
#         # Initial feature extraction
#         self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)
        
#         # Encoder (downsampling path)
#         self.encoder = nn.ModuleList()
#         channels = [base_channels, base_channels*2, base_channels*4]
        
#         # First downsampling block with attention
#         self.encoder.append(
#             nn.Sequential(
#                 ConvBlock(channels[0], channels[1], kernel_size=4, stride=2, padding=1),
#                 ResidualBlock(channels[1]),
#                 ResidualBlock(channels[1]),
#                 AttentionBlock(channels[1])
#             )
#         )
        
#         # Second downsampling block with attention
#         self.encoder.append(
#             nn.Sequential(
#                 ConvBlock(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
#                 ResidualBlock(channels[2]),
#                 ResidualBlock(channels[2]),
#                 AttentionBlock(channels[2])
#             )
#         )
        
#         # Bottleneck with attention
#         self.bottleneck = nn.Sequential(
#             ResidualBlock(channels[2]),
#             AttentionBlock(channels[2]),
#             ResidualBlock(channels[2]),
#             AttentionBlock(channels[2])
#         )
        
#         # Decoder (upsampling path)
#         self.decoder = nn.ModuleList()
        
#         # First upsampling block
#         self.decoder.append(
#             nn.Sequential(
#                 nn.ConvTranspose2d(channels[2], channels[1], kernel_size=4, stride=2, padding=1),
#                 nn.BatchNorm2d(channels[1]),
#                 nn.ReLU(inplace=True),
#                 ResidualBlock(channels[1]),
#                 AttentionBlock(channels[1])
#             )
#         )
        
#         # Second upsampling block
#         self.decoder.append(
#             nn.Sequential(
#                 nn.ConvTranspose2d(channels[1]*2, channels[0], kernel_size=4, stride=2, padding=1),  # *2 for skip connection
#                 nn.BatchNorm2d(channels[0]),
#                 nn.ReLU(inplace=True),
#                 ResidualBlock(channels[0]),
#                 AttentionBlock(channels[0])
#             )
#         )
        
#         # Final reconstruction
#         self.output_conv = nn.Sequential(
#             ConvBlock(channels[0]*2, channels[0], kernel_size=3, padding=1),  # *2 for skip connection
#             ConvBlock(channels[0], channels[0]//2, kernel_size=3, padding=1),
#             nn.Conv2d(channels[0]//2, in_channels, kernel_size=3, padding=1),
#             nn.Tanh()
#         )
        
#         # Additional guided filter for detail preservation
#         self.detail_branch = nn.Sequential(
#             ConvBlock(in_channels, 16, kernel_size=3, padding=1),
#             ConvBlock(16, 16, kernel_size=3, padding=1),
#             nn.Conv2d(16, 1, kernel_size=1, padding=0),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         # Extract guidance map for detail preservation
#         guidance = self.detail_branch(x)
        
#         # Initial features
#         features = [self.init_conv(x)]  # Store for skip connections
        
#         # Encoder
#         for enc_block in self.encoder:
#             features.append(enc_block(features[-1]))
        
#         # Bottleneck
#         bottle_features = self.bottleneck(features[-1])
        
#         # Decoder with skip connections
#         x1 = self.decoder[0](bottle_features)
#         x1 = torch.cat([x1, features[-2]], dim=1)  # Skip connection
        
#         x2 = self.decoder[1](x1)
#         x2 = torch.cat([x2, features[0]], dim=1)  # Skip connection
        
#         # Final output - predict residual
#         residual = self.output_conv(x2)
        
#         # Scale residual based on guidance (stronger modifications where fog is heavier)
#         weighted_residual = residual * guidance
        
#         # Apply to input
#         return torch.clamp(x + weighted_residual, 0, 1)
    
#     def get_info(self):
#         info = super().get_info()
#         info.update({
#             "model_type": "HighIntensityDehazeModel",
#             "base_channels": self.base_channels,
#             "n_blocks": self.n_blocks
#         })
#         return info

# class DualBranchAttentionModel(BaseDehazeModel):
#     """Complex model with dual branch processing for high-intensity fog"""
    
#     def __init__(self, in_channels=3, base_channels=96, n_blocks=9):
#         super(DualBranchAttentionModel, self).__init__()
        
#         self.in_channels = in_channels
#         self.base_channels = base_channels
#         self.n_blocks = n_blocks
        
#         # Global branch for overall structure
#         self.global_branch = nn.Sequential(
#             ConvBlock(in_channels, base_channels, kernel_size=7, padding=3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             ResidualBlock(base_channels),
#             AttentionBlock(base_channels),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             ResidualBlock(base_channels),
#             AttentionBlock(base_channels),
#             ResidualBlock(base_channels),
#             nn.UpsamplingBilinear2d(scale_factor=2),
#             ResidualBlock(base_channels),
#             nn.UpsamplingBilinear2d(scale_factor=2),
#             ConvBlock(base_channels, base_channels//2, kernel_size=3, padding=1)
#         )
        
#         # Local branch for detail preservation
#         self.local_branch = nn.Sequential(
#             ConvBlock(in_channels, base_channels//2, kernel_size=3, padding=1),
#             ResidualBlock(base_channels//2),
#             ResidualBlock(base_channels//2),
#             ConvBlock(base_channels//2, base_channels//2, kernel_size=3, padding=1)
#         )
        
#         # Transmission map estimation
#         self.transmission_branch = nn.Sequential(
#             ConvBlock(base_channels, base_channels//2, kernel_size=3, padding=1),
#             ConvBlock(base_channels//2, base_channels//4, kernel_size=3, padding=1),
#             nn.Conv2d(base_channels//4, 1, kernel_size=1, padding=0),
#             nn.Sigmoid()
#         )
        
#         # Final fusion
#         self.fusion_conv = nn.Sequential(
#             ConvBlock(base_channels, base_channels//2, kernel_size=3, padding=1),
#             nn.Conv2d(base_channels//2, in_channels, kernel_size=3, padding=1),
#             nn.Tanh()
#         )
        
#     def forward(self, x):
#         # Process through branches
#         global_features = self.global_branch(x)
#         local_features = self.local_branch(x)
        
#         # Concatenate features
#         concat_features = torch.cat([global_features, local_features], dim=1)
        
#         # Estimate transmission map (indicates fog density)
#         transmission = self.transmission_branch(concat_features)
        
#         # Generate final output
#         residual = self.fusion_conv(concat_features)
        
#         # Apply residual scaled by transmission map
#         # Lower transmission (denser fog) means stronger adjustments
#         return torch.clamp(x + (1 - transmission) * residual, 0, 1)
    
#     def get_info(self):
#         info = super().get_info()
#         info.update({
#             "model_type": "DualBranchAttentionModel",
#             "base_channels": self.base_channels,
#             "n_blocks": self.n_blocks
#         })
#         return info

# def create_high_intensity_model(config):
#     """Create dehazing model for high intensity fog based on config"""
#     model_type = config['dehazing']['high']['model_type']
    
#     if model_type == 'complex':
#         return HighIntensityDehazeModel(
#             base_channels=config['dehazing']['high']['channels'],
#             n_blocks=config['dehazing']['high']['blocks']
#         )
#     elif model_type == 'dual_branch':
#         return DualBranchAttentionModel(
#             base_channels=config['dehazing']['high']['channels'],
#             n_blocks=config['dehazing']['high']['blocks']
#         )
#     else:
#         # Use the versatile EncoderDecoder model with attention
#         return EncoderDecoder(
#             base_channels=config['dehazing']['high']['channels'],
#             n_blocks=config['dehazing']['high']['blocks'],
#             use_attention=True
#         )