import torch
import torch.nn as nn
from .base_model import BaseDehazeModel, ConvBlock, ResidualBlock, EncoderDecoder

class MediumIntensityDehazeModel(BaseDehazeModel):
    """Standard dehazing model for medium-intensity fog"""
    
    def __init__(self, in_channels=3, base_channels=64, n_blocks=6):
        super(MediumIntensityDehazeModel, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        
        # Initial feature extraction
        self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        channels = [base_channels, base_channels*2]
        
        # First downsampling block
        self.encoder.append(
            nn.Sequential(
                ConvBlock(channels[0], channels[1], kernel_size=4, stride=2, padding=1),
                ResidualBlock(channels[1]),
                ResidualBlock(channels[1])
            )
        )
        
        # Second downsampling block
        channels.append(base_channels*4)
        self.encoder.append(
            nn.Sequential(
                ConvBlock(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
                ResidualBlock(channels[2]),
                ResidualBlock(channels[2])
            )
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[2]),
            ResidualBlock(channels[2])
        )
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        
        # First upsampling block
        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(channels[2], channels[1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(inplace=True),
                ResidualBlock(channels[1])
            )
        )
        
        # Second upsampling block
        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(channels[1]*2, channels[0], kernel_size=4, stride=2, padding=1),  # *2 for skip connection
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True),
                ResidualBlock(channels[0])
            )
        )
        
        # Final reconstruction
        self.output_conv = nn.Sequential(
            ConvBlock(channels[0]*2, channels[0], kernel_size=3, padding=1),  # *2 for skip connection
            ConvBlock(channels[0], channels[0]//2, kernel_size=3, padding=1),
            nn.Conv2d(channels[0]//2, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
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
        
        # Scale to [-1, 1] range and add to input
        return torch.clamp(x + residual, 0, 1)
    
    def get_info(self):
        info = super().get_info()
        info.update({
            "model_type": "MediumIntensityDehazeModel",
            "base_channels": self.base_channels,
            "n_blocks": self.n_blocks
        })
        return info

class COrunInspiredModel(BaseDehazeModel):
    """Medium intensity dehazing model inspired by CORUN paper"""
    
    def __init__(self, in_channels=3, base_channels=64, n_blocks=6):
        super(COrunInspiredModel, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        
        # Initial feature extraction
        self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)
        
        # Multi-scale feature extraction
        self.scale1_conv = ConvBlock(base_channels, base_channels, kernel_size=3, padding=1)
        self.scale2_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.scale3_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            ConvBlock(base_channels, base_channels*4, kernel_size=3, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        
        # Feature fusion
        self.fusion_conv = ConvBlock(base_channels + base_channels*2 + base_channels*4, 
                                    base_channels*2, kernel_size=1, padding=0)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_channels*2) for _ in range(n_blocks)]
        )
        
        # Final reconstruction
        self.output_conv = nn.Sequential(
            ConvBlock(base_channels*2, base_channels, kernel_size=3, padding=1),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Initial features
        init_features = self.init_conv(x)
        
        # Multi-scale feature extraction
        scale1_features = self.scale1_conv(init_features)
        scale2_features = self.scale2_conv(init_features)
        scale3_features = self.scale3_conv(init_features)
        
        # Feature fusion
        fused_features = torch.cat([scale1_features, scale2_features, scale3_features], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        # Residual processing
        processed_features = self.residual_blocks(fused_features)
        
        # Final output - predict residual
        residual = self.output_conv(processed_features)
        
        # Scale to [-1, 1] range and add to input
        return torch.clamp(x + residual, 0, 1)
    
    def get_info(self):
        info = super().get_info()
        info.update({
            "model_type": "COrunInspiredModel",
            "base_channels": self.base_channels,
            "n_blocks": self.n_blocks
        })
        return info
    
def create_medium_intensity_model(config):
    """Create dehazing model for medium intensity fog based on config"""
    model_type = config['dehazing']['medium']['model_type']
    
    if model_type == 'corun':
        return COrunInspiredModel(
            base_channels=config['dehazing']['medium']['channels'],
            n_blocks=config['dehazing']['medium']['blocks']
        )
    else:
        # Always use the stable MediumIntensityDehazeModel instead of EncoderDecoder
        return MediumIntensityDehazeModel(
            base_channels=config['dehazing']['medium']['channels'],
            n_blocks=config['dehazing']['medium']['blocks']
        )


# import torch
# import torch.nn as nn
# from .base_model import BaseDehazeModel, ConvBlock, ResidualBlock, EncoderDecoder

# class MediumIntensityDehazeModel(BaseDehazeModel):
#     """Standard dehazing model for medium-intensity fog"""
    
#     def __init__(self, in_channels=3, base_channels=64, n_blocks=6):
#         super(MediumIntensityDehazeModel, self).__init__()
        
#         self.in_channels = in_channels
#         self.base_channels = base_channels
#         self.n_blocks = n_blocks
        
#         # Initial feature extraction
#         self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)
        
#         # Encoder (downsampling path)
#         self.encoder = nn.ModuleList()
#         channels = [base_channels, base_channels*2]
        
#         # First downsampling block
#         self.encoder.append(
#             nn.Sequential(
#                 ConvBlock(channels[0], channels[1], kernel_size=4, stride=2, padding=1),
#                 ResidualBlock(channels[1]),
#                 ResidualBlock(channels[1])
#             )
#         )
        
#         # Second downsampling block
#         channels.append(base_channels*4)
#         self.encoder.append(
#             nn.Sequential(
#                 ConvBlock(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
#                 ResidualBlock(channels[2]),
#                 ResidualBlock(channels[2])
#             )
#         )
        
#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             ResidualBlock(channels[2]),
#             ResidualBlock(channels[2])
#         )
        
#         # Decoder (upsampling path)
#         self.decoder = nn.ModuleList()
        
#         # First upsampling block
#         self.decoder.append(
#             nn.Sequential(
#                 nn.ConvTranspose2d(channels[2], channels[1], kernel_size=4, stride=2, padding=1),
#                 nn.BatchNorm2d(channels[1]),
#                 nn.ReLU(inplace=True),
#                 ResidualBlock(channels[1])
#             )
#         )
        
#         # Second upsampling block
#         self.decoder.append(
#             nn.Sequential(
#                 nn.ConvTranspose2d(channels[1]*2, channels[0], kernel_size=4, stride=2, padding=1),  # *2 for skip connection
#                 nn.BatchNorm2d(channels[0]),
#                 nn.ReLU(inplace=True),
#                 ResidualBlock(channels[0])
#             )
#         )
        
#         # Final reconstruction
#         self.output_conv = nn.Sequential(
#             ConvBlock(channels[0]*2, channels[0], kernel_size=3, padding=1),  # *2 for skip connection
#             ConvBlock(channels[0], channels[0]//2, kernel_size=3, padding=1),
#             nn.Conv2d(channels[0]//2, in_channels, kernel_size=3, padding=1),
#             nn.Tanh()
#         )
        
#     def forward(self, x):
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
        
#         # Scale to [-1, 1] range and add to input
#         return torch.clamp(x + residual, 0, 1)
    
#     def get_info(self):
#         info = super().get_info()
#         info.update({
#             "model_type": "MediumIntensityDehazeModel",
#             "base_channels": self.base_channels,
#             "n_blocks": self.n_blocks
#         })
#         return info

# class COrunInspiredModel(BaseDehazeModel):
#     """Medium intensity dehazing model inspired by CORUN paper"""
    
#     def __init__(self, in_channels=3, base_channels=64, n_blocks=6):
#         super(COrunInspiredModel, self).__init__()
        
#         self.in_channels = in_channels
#         self.base_channels = base_channels
#         self.n_blocks = n_blocks
        
#         # Initial feature extraction
#         self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)
        
#         # Multi-scale feature extraction
#         self.scale1_conv = ConvBlock(base_channels, base_channels, kernel_size=3, padding=1)
#         self.scale2_conv = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             ConvBlock(base_channels, base_channels*2, kernel_size=3, padding=1),
#             nn.UpsamplingBilinear2d(scale_factor=2)
#         )
#         self.scale3_conv = nn.Sequential(
#             nn.MaxPool2d(kernel_size=4, stride=4),
#             ConvBlock(base_channels, base_channels*4, kernel_size=3, padding=1),
#             nn.UpsamplingBilinear2d(scale_factor=4)
#         )
        
#         # Feature fusion
#         self.fusion_conv = ConvBlock(base_channels + base_channels*2 + base_channels*4, 
#                                     base_channels*2, kernel_size=1, padding=0)
        
#         # Residual blocks
#         self.residual_blocks = nn.Sequential(
#             *[ResidualBlock(base_channels*2) for _ in range(n_blocks)]
#         )
        
#         # Final reconstruction
#         self.output_conv = nn.Sequential(
#             ConvBlock(base_channels*2, base_channels, kernel_size=3, padding=1),
#             nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
#             nn.Tanh()
#         )
        
#     def forward(self, x):
#         # Initial features
#         init_features = self.init_conv(x)
        
#         # Multi-scale feature extraction
#         scale1_features = self.scale1_conv(init_features)
#         scale2_features = self.scale2_conv(init_features)
#         scale3_features = self.scale3_conv(init_features)
        
#         # Feature fusion
#         fused_features = torch.cat([scale1_features, scale2_features, scale3_features], dim=1)
#         fused_features = self.fusion_conv(fused_features)
        
#         # Residual processing
#         processed_features = self.residual_blocks(fused_features)
        
#         # Final output - predict residual
#         residual = self.output_conv(processed_features)
        
#         # Scale to [-1, 1] range and add to input
#         return torch.clamp(x + residual, 0, 1)
    
#     def get_info(self):
#         info = super().get_info()
#         info.update({
#             "model_type": "COrunInspiredModel",
#             "base_channels": self.base_channels,
#             "n_blocks": self.n_blocks
#         })
#         return info
    
# def create_medium_intensity_model(config):
#     """Create dehazing model for medium intensity fog based on config"""
#     model_type = config['dehazing']['medium']['model_type']
    
#     if model_type == 'standard':
#         return MediumIntensityDehazeModel(
#             base_channels=config['dehazing']['medium']['channels'],
#             n_blocks=config['dehazing']['medium']['blocks']
#         )
#     elif model_type == 'corun':
#         return COrunInspiredModel(
#             base_channels=config['dehazing']['medium']['channels'],
#             n_blocks=config['dehazing']['medium']['blocks']
#         )
#     else:
#         # Use the versatile EncoderDecoder model
#         return EncoderDecoder(
#             base_channels=config['dehazing']['medium']['channels'],
#             n_blocks=config['dehazing']['medium']['blocks'],
#             use_attention=False
#         )