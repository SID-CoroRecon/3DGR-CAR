import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize


def create_centerline_mask(projection: torch.Tensor, sigma: float = 1.0, threshold: float = 0.1) -> torch.Tensor:
    """
    Create binary centerline mask by skeletonizing and binarizing 2D projection images.
    
    Args:
        projection: Input projection [B, H, W] or [B, 1, H, W]
        sigma: Gaussian smoothing parameter
        threshold: Binarization threshold
        
    Returns:
        centerline_mask: Binary centerline mask [B, H, W]
    """
    if projection.dim() == 4:
        projection = projection.squeeze(1)  # Remove channel dimension if present
    
    batch_size = projection.shape[0]
    device = projection.device
    
    centerline_masks = []
    
    for i in range(batch_size):
        # Convert to numpy for skeletonization
        proj_np = projection[i].detach().cpu().numpy()
        
        # Apply Gaussian smoothing
        proj_smooth = gaussian_filter(proj_np, sigma=sigma)
        
        # Binarize
        binary_proj = (proj_smooth > threshold).astype(np.uint8)
        
        # Skeletonize to get centerline
        if np.any(binary_proj):
            skeleton = skeletonize(binary_proj)
        else:
            skeleton = binary_proj
        
        # Convert back to tensor
        centerline_mask = torch.from_numpy(skeleton.astype(np.float32)).to(device)
        centerline_masks.append(centerline_mask)
    
    return torch.stack(centerline_masks, dim=0)


class CombinedGaussianLoss(nn.Module):
    """
    Combined loss function for Gaussian reconstruction as described in the paper.
    
    L_G = Σ_i (α * L_L2 + (1-α) * L_clL2)
    
    where:
    - L_L2 = ||P̂_i - P_i||²₂ (projection loss)
    - L_clL2 = ||(P̂_i - P_i) * M_cl||²₂ (projected vessel centerline loss)
    - M_cl is the binary mask obtained by skeletonizing and binarizing 2D projection images
    - α = 0.5 (set empirically)
    """
    
    def __init__(self, alpha: float = 0.5, centerline_sigma: float = 1.0, 
                 centerline_threshold: float = 0.1):
        super(CombinedGaussianLoss, self).__init__()
        self.alpha = alpha
        self.centerline_sigma = centerline_sigma
        self.centerline_threshold = centerline_threshold
        
    def forward(self, pred_projections: torch.Tensor, gt_projections: torch.Tensor) -> tuple:
        """
        Compute combined Gaussian reconstruction loss.
        
        Args:
            pred_projections: Predicted projections [B, N_proj, H, W]
            gt_projections: Ground truth projections [B, N_proj, H, W]
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual loss components
        """
        batch_size, num_proj = pred_projections.shape[:2]
        
        # Initialize loss accumulators
        total_l2_loss = 0.0
        total_cl_loss = 0.0
        
        # Compute loss for each projection
        for i in range(num_proj):
            pred_proj_i = pred_projections[:, i]  # [B, H, W]
            gt_proj_i = gt_projections[:, i]      # [B, H, W]
            
            # 1. Standard L2 projection loss
            l2_loss = F.mse_loss(pred_proj_i, gt_proj_i)
            
            # 2. Centerline-weighted L2 loss
            # Create centerline mask from ground truth projection
            centerline_mask = create_centerline_mask(
                gt_proj_i, 
                sigma=self.centerline_sigma,
                threshold=self.centerline_threshold
            )
            
            # Compute difference
            proj_diff = pred_proj_i - gt_proj_i
            
            # Apply centerline mask and compute weighted L2 loss
            weighted_diff = proj_diff * centerline_mask
            cl_loss = torch.mean(weighted_diff ** 2)
            
            # Accumulate losses
            total_l2_loss += l2_loss
            total_cl_loss += cl_loss
        
        # Average over projections
        avg_l2_loss = total_l2_loss / num_proj
        avg_cl_loss = total_cl_loss / num_proj
        
        # Combine losses with alpha weighting
        total_loss = self.alpha * avg_l2_loss + (1 - self.alpha) * avg_cl_loss
        
        # Create loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'l2_loss': avg_l2_loss.item(),
            'centerline_loss': avg_cl_loss.item(),
            'alpha': self.alpha
        }
        
        return total_loss, loss_dict


class GaussianReconstructionLoss(nn.Module):
    """
    Enhanced loss function with additional regularization terms.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.1, gamma: float = 0.01):
        super(GaussianReconstructionLoss, self).__init__()
        self.combined_loss = CombinedGaussianLoss(alpha=alpha)
        self.beta = beta    # Weight for density regularization
        self.gamma = gamma  # Weight for smoothness regularization
        
    def forward(self, pred_projections: torch.Tensor, gt_projections: torch.Tensor,
                gaussians_model=None) -> tuple:
        """
        Compute enhanced Gaussian reconstruction loss.
        
        Args:
            pred_projections: Predicted projections [B, N_proj, H, W]
            gt_projections: Ground truth projections [B, N_proj, H, W]
            gaussians_model: Gaussian model for regularization (optional)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual loss components
        """
        # Main combined loss
        main_loss, loss_dict = self.combined_loss(pred_projections, gt_projections)
        
        total_loss = main_loss
        
        # Add regularization terms if Gaussian model is provided
        if gaussians_model is not None:
            # Density regularization: encourage sparse but meaningful Gaussians
            if hasattr(gaussians_model, '_density') and gaussians_model._density is not None:
                density_reg = torch.mean(gaussians_model._density ** 2)
                total_loss += self.beta * density_reg
                loss_dict['density_reg'] = density_reg.item()
            
            # Smoothness regularization: encourage smooth Gaussian distributions
            if hasattr(gaussians_model, '_xyz') and gaussians_model._xyz is not None:
                # Compute smoothness penalty on Gaussian positions
                xyz = gaussians_model._xyz
                if xyz.shape[0] > 1:
                    # Compute pairwise distances and encourage local smoothness
                    distances = torch.cdist(xyz, xyz)
                    # Penalize very close Gaussians (avoid redundancy)
                    close_penalty = torch.sum(torch.exp(-distances * 10))
                    total_loss += self.gamma * close_penalty
                    loss_dict['smoothness_reg'] = close_penalty.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = None) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between predicted and target images.
    
    Args:
        pred: Predicted images [B, ...]
        target: Target images [B, ...]
        max_val: Maximum possible value (auto-computed if None)
        
    Returns:
        psnr: PSNR value in dB
    """
    if max_val is None:
        max_val = target.max().item()
    
    mse = F.mse_loss(pred, target)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr.item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, 
                 window_size: int = 11, sigma: float = 1.5) -> float:
    """
    Compute Structural Similarity Index (SSIM) between predicted and target images.
    Simplified implementation for 2D images.
    
    Args:
        pred: Predicted images [B, H, W]
        target: Target images [B, H, W]
        window_size: Size of the sliding window
        sigma: Standard deviation of Gaussian window
        
    Returns:
        ssim: SSIM value
    """
    # Constants for SSIM
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    window = g.outer(g).unsqueeze(0).unsqueeze(0)  # [1, 1, window_size, window_size]
    
    # Compute local means
    mu1 = F.conv2d(pred.unsqueeze(1), window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(target.unsqueeze(1), window, padding=window_size//2, groups=1)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.conv2d(pred.unsqueeze(1) ** 2, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(target.unsqueeze(1) ** 2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(pred.unsqueeze(1) * target.unsqueeze(1), window, padding=window_size//2) - mu1_mu2
    
    # Compute SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    return ssim_map.mean().item()


# Test the loss functions
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    batch_size = 2
    num_proj = 2
    height, width = 128, 128
    
    # Create test projections
    pred_projections = torch.rand(batch_size, num_proj, height, width).to(device)
    gt_projections = torch.rand(batch_size, num_proj, height, width).to(device)
    
    # Make ground truth more vessel-like (higher values in center)
    for b in range(batch_size):
        for p in range(num_proj):
            # Add some vessel-like structures
            center_y, center_x = height // 2, width // 2
            y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
            y, x = y.float().to(device), x.float().to(device)
            
            # Create curved vessel
            dist = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
            vessel = torch.exp(-(dist - 20) ** 2 / 100) * 0.8
            gt_projections[b, p] += vessel
    
    # Test combined loss
    print("Testing Combined Gaussian Loss...")
    combined_loss = CombinedGaussianLoss().to(device)
    
    loss_val, loss_dict = combined_loss(pred_projections, gt_projections)
    
    print(f"Total loss: {loss_val.item():.6f}")
    print("Loss breakdown:", loss_dict)
    
    # Test PSNR computation
    psnr = compute_psnr(pred_projections, gt_projections)
    print(f"PSNR: {psnr:.2f} dB")
    
    # Test SSIM computation
    ssim = compute_ssim(pred_projections[:, 0], gt_projections[:, 0])
    print(f"SSIM: {ssim:.4f}")
    
    print("Loss functions test completed successfully!")