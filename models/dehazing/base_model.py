import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_bn=True, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        if activation is not None:
            layers.append(activation)
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers"""
    
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = ConvBlock(channels, channels, kernel_size, padding=kernel_size//2)
        self.conv2 = ConvBlock(channels, channels, kernel_size, padding=kernel_size//2, activation=None)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual  # Skip connection
        return self.relu(out)

class AttentionBlock(nn.Module):
    """Channel and spatial attention module"""
    
    def __init__(self, channels, reduction=16):
        super(AttentionBlock, self).__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        
        x = x * channel_out
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.sigmoid(self.conv_spatial(spatial_in))
        
        return x * spatial_out

class BaseDehazeModel(nn.Module):
    """Base class for all dehazing models"""
    
    def __init__(self):
        super(BaseDehazeModel, self).__init__()
    
    def forward(self, x):
        """Should be implemented by subclasses"""
        raise NotImplementedError
    
    def get_info(self):
        """Return model info dictionary"""
        return {
            "model_type": self.__class__.__name__,
            "params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

class EncoderDecoder(BaseDehazeModel):
    """Encoder-Decoder architecture for image dehazing"""
    
    def __init__(self, in_channels=3, base_channels=64, n_blocks=6, use_attention=False):
        super(EncoderDecoder, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        self.use_attention = use_attention
        
        # Initial feature extraction
        self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(3):  # 3 downsampling blocks
            # Downsample
            self.encoder_blocks.append(
                ConvBlock(current_channels, current_channels*2, kernel_size=4, stride=2, padding=1)
            )
            current_channels *= 2
            
            # Add residual blocks
            for _ in range(n_blocks // 3):
                self.encoder_blocks.append(ResidualBlock(current_channels))
        
        # Bottleneck blocks with optional attention
        self.bottleneck = nn.ModuleList()
        for _ in range(2):
            self.bottleneck.append(ResidualBlock(current_channels))
            
        if use_attention:
            self.bottleneck.append(AttentionBlock(current_channels))
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(3):  # 3 upsampling blocks
            # Add residual blocks
            for _ in range(n_blocks // 3):
                self.decoder_blocks.append(ResidualBlock(current_channels))
            
            # Upsample
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_channels, current_channels//2, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(current_channels//2),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels //= 2
        
        # Final output layer
        self.output_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels, kernel_size=3, padding=1),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Initial feature extraction
        features = self.init_conv(x)
        
        # Store encoder features for skip connections
        skip_connections = [features]
        
        # Encoder
        for block in self.encoder_blocks:
            features = block(features)
            if isinstance(block, ResidualBlock):
                skip_connections.append(features)
        
        # Bottleneck
        for block in self.bottleneck:
            features = block(features)
        
        # Decoder with skip connections
        skip_idx = len(skip_connections) - 1
        
        for block in self.decoder_blocks:
            # Apply the decoder block
            features = block(features)
            
            # Handle skip connections after upsampling
            if isinstance(block, nn.Sequential) and skip_idx >= 0:
                # Ensure spatial dimensions match
                if features.shape[2:] != skip_connections[skip_idx].shape[2:]:
                    features = nn.functional.interpolate(
                        features, 
                        size=skip_connections[skip_idx].shape[2:],
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Instead of adding (which requires same channels),
                # we concatenate features along channel dimension
                features = torch.cat([features, skip_connections[skip_idx]], dim=1)
                
                # Apply a 1x1 conv to get back to the expected channel count
                # We'll use a temporary conv layer for this
                temp_conv = nn.Conv2d(
                    features.shape[1], 
                    features.shape[1] // 2,  # Reduce channels by half
                    kernel_size=1,
                    padding=0
                ).to(features.device)
                
                features = temp_conv(features)
                skip_idx -= 1
        
        # Final output - we predict the residual (clean - hazy)
        residual = self.output_conv(features)
        
        # Scale to [-1, 1] range and add to input
        return torch.clamp(x + residual, 0, 1)
        
        # Final output - we predict the residual (clean - hazy)
        residual = self.output_conv(features)
        
        # Scale to [-1, 1] range and add to input
        return torch.clamp(x + residual, 0, 1)
    
    def get_info(self):
        info = super().get_info()
        info.update({
            "base_channels": self.base_channels,
            "n_blocks": self.n_blocks,
            "use_attention": self.use_attention
        })
        return info


# import torch
# import torch.nn as nn

# class ConvBlock(nn.Module):
#     """Basic convolutional block with BatchNorm and activation"""
    
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
#                  use_bn=True, activation=nn.ReLU()):
#         super(ConvBlock, self).__init__()
        
#         layers = [
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
#         ]
        
#         if use_bn:
#             layers.append(nn.BatchNorm2d(out_channels))
            
#         if activation is not None:
#             layers.append(activation)
            
#         self.block = nn.Sequential(*layers)
        
#     def forward(self, x):
#         return self.block(x)

# class ResidualBlock(nn.Module):
#     """Residual block with two convolutional layers"""
    
#     def __init__(self, channels, kernel_size=3):
#         super(ResidualBlock, self).__init__()
        
#         self.conv1 = ConvBlock(channels, channels, kernel_size, padding=kernel_size//2)
#         self.conv2 = ConvBlock(channels, channels, kernel_size, padding=kernel_size//2, activation=None)
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out += residual  # Skip connection
#         return self.relu(out)

# class AttentionBlock(nn.Module):
#     """Channel and spatial attention module"""
    
#     def __init__(self, channels, reduction=16):
#         super(AttentionBlock, self).__init__()
        
#         # Channel attention
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels, 1, bias=False)
#         )
        
#         self.sigmoid = nn.Sigmoid()
        
#         # Spatial attention
#         self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
#     def forward(self, x):
#         # Channel attention
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         channel_out = self.sigmoid(avg_out + max_out)
        
#         x = x * channel_out
        
#         # Spatial attention
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         spatial_in = torch.cat([avg_out, max_out], dim=1)
#         spatial_out = self.sigmoid(self.conv_spatial(spatial_in))
        
#         return x * spatial_out

# class BaseDehazeModel(nn.Module):
#     """Base class for all dehazing models"""
    
#     def __init__(self):
#         super(BaseDehazeModel, self).__init__()
    
#     def forward(self, x):
#         """Should be implemented by subclasses"""
#         raise NotImplementedError
    
#     def get_info(self):
#         """Return model info dictionary"""
#         return {
#             "model_type": self.__class__.__name__,
#             "params": sum(p.numel() for p in self.parameters()),
#             "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
#         }

# class EncoderDecoder(BaseDehazeModel):
#     """Encoder-Decoder architecture for image dehazing"""
    
#     def __init__(self, in_channels=3, base_channels=64, n_blocks=6, use_attention=False):
#         super(EncoderDecoder, self).__init__()
        
#         self.in_channels = in_channels
#         self.base_channels = base_channels
#         self.n_blocks = n_blocks
#         self.use_attention = use_attention
        
#         # Initial feature extraction
#         self.init_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)
        
#         # Encoder blocks
#         self.encoder_blocks = nn.ModuleList()
#         current_channels = base_channels
        
#         for i in range(3):  # 3 downsampling blocks
#             # Downsample
#             self.encoder_blocks.append(
#                 ConvBlock(current_channels, current_channels*2, kernel_size=4, stride=2, padding=1)
#             )
#             current_channels *= 2
            
#             # Add residual blocks
#             for _ in range(n_blocks // 3):
#                 self.encoder_blocks.append(ResidualBlock(current_channels))
        
#         # Bottleneck blocks with optional attention
#         self.bottleneck = nn.ModuleList()
#         for _ in range(2):
#             self.bottleneck.append(ResidualBlock(current_channels))
            
#         if use_attention:
#             self.bottleneck.append(AttentionBlock(current_channels))
        
#         # Decoder blocks
#         self.decoder_blocks = nn.ModuleList()
        
#         for i in range(3):  # 3 upsampling blocks
#             # Add residual blocks
#             for _ in range(n_blocks // 3):
#                 self.decoder_blocks.append(ResidualBlock(current_channels))
            
#             # Upsample
#             self.decoder_blocks.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(current_channels, current_channels//2, kernel_size=4, stride=2, padding=1),
#                     nn.BatchNorm2d(current_channels//2),
#                     nn.ReLU(inplace=True)
#                 )
#             )
#             current_channels //= 2
        
#         # Final output layer
#         self.output_conv = nn.Sequential(
#             ConvBlock(base_channels, base_channels, kernel_size=3, padding=1),
#             nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
#             nn.Tanh()
#         )
        
#     def forward(self, x):
#         # Initial feature extraction
#         features = self.init_conv(x)
        
#         # Store encoder features for skip connections
#         skip_connections = [features]
        
#         # Encoder
#         for block in self.encoder_blocks:
#             features = block(features)
#             if isinstance(block, ResidualBlock):
#                 skip_connections.append(features)
        
#         # Bottleneck
#         for block in self.bottleneck:
#             features = block(features)
        
#         # Decoder with skip connections
#         skip_idx = len(skip_connections) - 1
        
#         for block in self.decoder_blocks:
#             if isinstance(block, nn.Sequential) and skip_idx >= 0:  # After upsampling, add skip connection
#                 features = block(features)
#                 features = features + skip_connections[skip_idx]
#                 skip_idx -= 1
#             else:
#                 features = block(features)
        
#         # Final output - we predict the residual (clean - hazy)
#         residual = self.output_conv(features)
        
#         # Scale to [-1, 1] range and add to input
#         return torch.clamp(x + residual, 0, 1)
    
#     def get_info(self):
#         info = super().get_info()
#         info.update({
#             "base_channels": self.base_channels,
#             "n_blocks": self.n_blocks,
#             "use_attention": self.use_attention
#         })
#         return info