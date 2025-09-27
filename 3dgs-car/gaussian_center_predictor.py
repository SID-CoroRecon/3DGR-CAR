import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GaussianCenterPredictor(nn.Module):
    """
    Gaussian Center Predictor Network (GCP)
    
    Takes a single H×W×1 grayscale projection image as input and outputs 
    positional parameters M = (d, Δx, Δy, Δz) for Gaussian centers.
    
    Architecture: U-Net based network
    Input: 128×128×1 projection image
    Output: (H/α)×(W/α)×4 tensor where α is downsampling factor
    """
    
    def __init__(self, input_channels=1, output_channels=4, downsampling_factor=2, bilinear=False):
        super(GaussianCenterPredictor, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels  # 4 channels for (d, Δx, Δy, Δz)
        self.downsampling_factor = downsampling_factor
        self.bilinear = bilinear

        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Final 1x1 convolution to get 4 output channels
        self.outc = OutConv(64, output_channels)
        
        # Average pooling for downsampling if needed
        if downsampling_factor > 1:
            self.downsample = nn.AvgPool2d(kernel_size=downsampling_factor, stride=downsampling_factor)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input projection image [B, 1, H, W]
            
        Returns:
            output: Positional parameters [B, 4, H/α, W/α] where 4 = (d, Δx, Δy, Δz)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final convolution to get 4 channels
        logits = self.outc(x)
        
        # Apply downsampling if specified
        output = self.downsample(logits)
        
        return output

    def predict_gaussian_centers(self, x, camera_params=None):
        """
        Predict 3D Gaussian center positions from a single projection
        
        Args:
            x: Input projection image [B, 1, H, W]
            camera_params: Camera parameters for depth conversion (optional)
            
        Returns:
            centers: 3D positions [B, N, 3] where N = (H/α) * (W/α)
            depths: Depth values [B, N]
            offsets: 3D offsets [B, N, 3]
        """
        # Get positional parameters [B, 4, H/α, W/α]
        params = self.forward(x)
        
        B, C, H, W = params.shape
        N = H * W  # Number of predicted centers
        
        # Reshape to [B, N, 4]
        params = params.permute(0, 2, 3, 1).contiguous().view(B, N, 4)
        
        # Split into depth and offsets
        depth = params[:, :, 0]  # [B, N]
        offsets = params[:, :, 1:4]  # [B, N, 3] for (Δx, Δy, Δz)
        
        # Apply activation functions
        depth = torch.sigmoid(depth) * 100.0  # Scale depth to reasonable range
        offsets = torch.tanh(offsets) * 10.0   # Scale offsets
        
        # Generate base grid coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        
        # Create base coordinates [H, W, 2]
        base_coords = torch.stack([x_coords, y_coords], dim=-1)
        base_coords = base_coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
        base_coords = base_coords.view(B, N, 2)  # [B, N, 2]
        
        # Combine base coordinates with depth to get 3D positions
        z_coords = depth.unsqueeze(-1)  # [B, N, 1]
        base_3d = torch.cat([base_coords, z_coords], dim=-1)  # [B, N, 3]
        
        # Add offsets to get final center positions
        centers = base_3d + offsets  # [B, N, 3]
        
        return centers, depth, offsets


def create_gcp_network(downsampling_factor=2):
    """
    Create a Gaussian Center Predictor network
    
    Args:
        input_size: Input image size (H, W)
        downsampling_factor: Factor to reduce number of initial Gaussians
        
    Returns:
        model: GaussianCenterPredictor instance
    """
    model = GaussianCenterPredictor(
        input_channels=1,
        output_channels=4,
        downsampling_factor=downsampling_factor,
        bilinear=True
    )
    return model


# Test function
if __name__ == "__main__":
    # Test the network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_gcp_network(downsampling_factor=2)
    model = model.to(device)
    
    # Test input
    batch_size = 2
    test_input = torch.randn(batch_size, 1, 128, 128).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
        centers, depths, offsets = model.predict_gaussian_centers(test_input)
        
    print(f"Raw output shape: {output.shape}")
    print(f"Predicted centers shape: {centers.shape}")
    print(f"Predicted depths shape: {depths.shape}")
    print(f"Predicted offsets shape: {offsets.shape}")
    
    print("GCP Network test completed successfully!")