import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def chamfer_distance_loss(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    """
    Compute Chamfer Distance Loss between predicted and ground truth point sets.
    
    Args:
        pred_points: Predicted points [B, N1, 3]
        gt_points: Ground truth points [B, N2, 3]
        
    Returns:
        loss: Chamfer distance loss
    """
    # Compute pairwise distances
    # pred_points: [B, N1, 3] -> [B, N1, 1, 3]
    # gt_points: [B, N2, 3] -> [B, 1, N2, 3]
    pred_expanded = pred_points.unsqueeze(2)  # [B, N1, 1, 3]
    gt_expanded = gt_points.unsqueeze(1)      # [B, 1, N2, 3]
    
    # Compute squared distances [B, N1, N2]
    distances = torch.sum((pred_expanded - gt_expanded) ** 2, dim=3)
    
    # Find minimum distances
    min_pred_to_gt, _ = torch.min(distances, dim=2)  # [B, N1]
    min_gt_to_pred, _ = torch.min(distances, dim=1)  # [B, N2]
    
    # Compute Chamfer distance
    loss = torch.mean(min_pred_to_gt) + torch.mean(min_gt_to_pred)
    
    return loss


def soft_cldice_loss(pred_volume: torch.Tensor, gt_volume: torch.Tensor, 
                     smooth: float = 1e-5) -> torch.Tensor:
    """
    Compute Soft-clDice Loss for tubular structures.
    
    Args:
        pred_volume: Predicted volume [B, 1, D, H, W]
        gt_volume: Ground truth volume [B, 1, D, H, W]
        smooth: Smoothing factor
        
    Returns:
        loss: Soft-clDice loss
    """
    # Apply softmax to get probabilities
    pred_prob = torch.sigmoid(pred_volume).float()
    gt_prob = gt_volume.float()
    
    # Compute soft skeletonization (approximation)
    # For simplicity, we use morphological operations
    pred_skel = soft_skeletonize(pred_prob)
    gt_skel = soft_skeletonize(gt_prob)
    
    # Compute intersection and union
    intersection = torch.sum(pred_skel * gt_skel, dim=(2, 3, 4))
    pred_sum = torch.sum(pred_skel, dim=(2, 3, 4))
    gt_sum = torch.sum(gt_skel, dim=(2, 3, 4))
    
    # Compute soft-clDice
    dice = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)
    loss = 1.0 - torch.mean(dice)
    
    return loss


def soft_skeletonize(volume: torch.Tensor) -> torch.Tensor:
    """
    Soft approximation of skeletonization using morphological operations.
    """
    # Simple approximation: use erosion to get centerline-like structure
    kernel_size = 3
    padding = kernel_size // 2
    
    # Apply erosion followed by dilation (opening operation)
    eroded = F.max_pool3d(-volume, kernel_size, stride=1, padding=padding)
    eroded = -eroded
    
    return eroded


def silog_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, 
               alpha: float = 0.1, beta: float = 10.0) -> torch.Tensor:
    """
    Scale-Invariant Logarithmic Loss for depth prediction.
    
    Args:
        pred_depth: Predicted depth [B, H, W]
        gt_depth: Ground truth depth [B, H, W]
        alpha: Small constant to avoid log(0)
        beta: Weight for mean term
        
    Returns:
        loss: SILog loss
    """
    # Add small constant to avoid log(0)
    pred_log = torch.log(pred_depth + alpha)
    gt_log = torch.log(gt_depth + alpha)
    
    # Compute difference
    g = pred_log - gt_log
    
    # Compute variance and mean
    g_var = torch.var(g.view(g.size(0), -1), dim=1, unbiased=False)
    g_mean = torch.mean(g.view(g.size(0), -1), dim=1)
    
    # SILog loss
    loss = 10 * torch.mean(g_var) + beta * torch.mean(g_mean ** 2)
    
    return loss


def gradient_l1_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
    """
    L1 loss on depth gradients.
    
    Args:
        pred_depth: Predicted depth [B, H, W]
        gt_depth: Ground truth depth [B, H, W]
        
    Returns:
        loss: Gradient L1 loss
    """
    # Compute gradients
    pred_grad_x = torch.abs(pred_depth[:, :-1, :] - pred_depth[:, 1:, :])
    pred_grad_y = torch.abs(pred_depth[:, :, :-1] - pred_depth[:, :, 1:])
    
    gt_grad_x = torch.abs(gt_depth[:, :-1, :] - gt_depth[:, 1:, :])
    gt_grad_y = torch.abs(gt_depth[:, :, :-1] - gt_depth[:, :, 1:])
    
    # L1 loss on gradients
    loss_x = F.l1_loss(pred_grad_x, gt_grad_x)
    loss_y = F.l1_loss(pred_grad_y, gt_grad_y)
    
    return loss_x + loss_y


def masked_l1_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, 
                   mask: torch.Tensor) -> torch.Tensor:
    """
    Masked L1 loss for depth prediction.
    
    Args:
        pred_depth: Predicted depth [B, H, W]
        gt_depth: Ground truth depth [B, H, W]
        mask: Binary mask [B, H, W]
        
    Returns:
        loss: Masked L1 loss
    """
    diff = torch.abs(pred_depth - gt_depth) * mask
    loss = torch.sum(diff) / (torch.sum(mask) + 1e-8)
    
    return loss


def depth_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Combined depth loss function.
    
    Args:
        pred_depth: Predicted depth [B, H, W]
        gt_depth: Ground truth depth [B, H, W]
        mask: Binary mask [B, H, W] (optional)
        
    Returns:
        loss: Combined depth loss
    """
    # SILog loss
    silog = silog_loss(pred_depth, gt_depth)
    
    # Gradient L1 loss
    grad_l1 = gradient_l1_loss(pred_depth, gt_depth)
    
    # Masked L1 loss
    if mask is not None:
        mask_l1 = masked_l1_loss(pred_depth, gt_depth, mask)
    else:
        mask_l1 = F.l1_loss(pred_depth, gt_depth)
    
    # Combine losses
    total_loss = silog + grad_l1 + mask_l1
    
    return total_loss


class GCPLoss(nn.Module):
    """
    Combined loss function for Gaussian Center Predictor training.
    
    L(C) = γ1 * L_cham + γ2 * L_clDice + γ3 * L_depth
    """
    
    def __init__(self, gamma1_base: float = 2.0, gamma2: float = 0.5, gamma3: float = 0.01):
        super(GCPLoss, self).__init__()
        self.gamma1_base = gamma1_base
        self.gamma2 = gamma2
        self.gamma3 = gamma3
    
    def forward(self, pred_centers: torch.Tensor, pred_volume: torch.Tensor, 
                pred_depth: torch.Tensor, gt_points: torch.Tensor, 
                gt_volume: torch.Tensor, gt_depth: torch.Tensor,
                depth_mask: Optional[torch.Tensor] = None,
                iteration: int = 1) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined GCP loss.
        
        Args:
            pred_centers: Predicted 3D centers [B, N, 3]
            pred_volume: Predicted volume [B, 1, D, H, W]
            pred_depth: Predicted depth [B, H, W]
            gt_points: Ground truth 3D points [B, M, 3]
            gt_volume: Ground truth volume [B, 1, D, H, W]
            gt_depth: Ground truth depth [B, H, W]
            depth_mask: Depth mask [B, H, W] (optional)
            iteration: Current training iteration
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Dynamic weight for Chamfer loss
        gamma1 = self.gamma1_base * np.log(max(iteration, 1) / 20000.0)
        
        # Chamfer distance loss
        chamfer_loss = chamfer_distance_loss(pred_centers, gt_points)
        
        # Soft-clDice loss
        cldice_loss = soft_cldice_loss(pred_volume, gt_volume)
        
        # Depth loss
        depth_loss_val = depth_loss(pred_depth, gt_depth, depth_mask)
        
        # Combine losses
        total_loss = gamma1 * chamfer_loss + self.gamma2 * cldice_loss + self.gamma3 * depth_loss_val
        
        # Create loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'chamfer_loss': chamfer_loss.item(),
            'cldice_loss': cldice_loss.item(),
            'depth_loss': depth_loss_val.item(),
            'gamma1': gamma1,
            'gamma2': self.gamma2,
            'gamma3': self.gamma3
        }
        
        return total_loss, loss_dict


# Test the loss functions
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    batch_size = 2
    num_pred_points = 1000
    num_gt_points = 800
    
    # Test Chamfer distance
    pred_points = torch.randn(batch_size, num_pred_points, 3).to(device)
    gt_points = torch.randn(batch_size, num_gt_points, 3).to(device)
    
    chamfer_loss = chamfer_distance_loss(pred_points, gt_points)
    print(f"Chamfer loss: {chamfer_loss.item():.6f}")
    
    # Test depth loss
    pred_depth = torch.rand(batch_size, 64, 64).to(device) * 10
    gt_depth = torch.rand(batch_size, 64, 64).to(device) * 10
    
    depth_loss_val = depth_loss(pred_depth, gt_depth)
    print(f"Depth loss: {depth_loss_val.item():.6f}")
    
    # Test soft-clDice loss
    pred_volume = torch.rand(batch_size, 1, 32, 32, 32).to(device)
    gt_volume = torch.rand(batch_size, 1, 32, 32, 32).to(device)
    
    cldice_loss = soft_cldice_loss(pred_volume, gt_volume)
    print(f"Soft-clDice loss: {cldice_loss.item():.6f}")
    
    # Test combined loss
    gcp_loss = GCPLoss().to(device)
    total_loss, loss_dict = gcp_loss(
        pred_points, pred_volume, pred_depth,
        gt_points, gt_volume, gt_depth,
        iteration=1000
    )
    
    print(f"Total GCP loss: {total_loss.item():.6f}")
    print("Loss breakdown:", loss_dict)
    
    print("GCP loss functions test completed successfully!")