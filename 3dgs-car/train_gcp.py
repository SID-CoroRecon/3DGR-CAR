import os
import json
import glob
import torch
import argparse
import numpy as np
import torch.optim as optim
from odl.contrib import torch as odl_torch
from torch.utils.data import Dataset, DataLoader

from gcp_losses import GCPLoss
from ct_geometry_projector import ConeBeam3DProjector, build_conebeam_gemotry, Initialization_ConeBeam
from gaussian_center_predictor import create_gcp_network


def voxelize_point_cloud(points, volume_shape, device, intensity=1.0, sigma=1.0):
    """
    Convert 3D point cloud to voxelized volume with Gaussian smoothing.
    
    Args:
        points: Tensor of shape [B, N, 3] containing 3D point coordinates
        volume_shape: Target volume shape [B, C, D, H, W]
        device: Device to use for computation
        intensity: Base intensity value to assign to voxels
        sigma: Standard deviation for Gaussian smoothing
    
    Returns:
        Voxelized volume tensor of shape matching volume_shape
    """
    batch_size, channels, depth, height, width = volume_shape
    
    # Initialize empty volume
    pred_volume = torch.zeros(volume_shape, device=device, dtype=torch.float32)
    
    for b in range(batch_size):
        if points[b].shape[0] == 0:
            continue
            
        # Get points for this batch item
        batch_points = points[b]  # [N, 3]
        
        # Normalize points to volume coordinate system
        # GCP outputs: x,y in [-1, 1] range, z (depth) in [0, 100] range
        # Convert to voxel coordinates [0, volume_size-1]
        
        # Scale x, y from [-1, 1] to [0, volume_size-1]
        scaled_points = batch_points.clone()
        scaled_points[:, 0] = (batch_points[:, 0] + 1.0) * 0.5 * (width - 1)   # x: [-1,1] -> [0, W-1]
        scaled_points[:, 1] = (batch_points[:, 1] + 1.0) * 0.5 * (height - 1)  # y: [-1,1] -> [0, H-1]
        
        # Scale z (depth) from [0, 100] to [0, D-1]
        # Assume depth range [0, 100] maps to full volume depth
        scaled_points[:, 2] = torch.clamp(batch_points[:, 2] / 100.0 * (depth - 1), 0, depth - 1)
        
        batch_points = scaled_points
        
        # Convert continuous coordinates to discrete voxel indices
        voxel_coords = torch.round(batch_points).long()
        
        # Clamp coordinates to valid range
        voxel_coords[:, 0] = torch.clamp(voxel_coords[:, 0], 0, depth - 1)
        voxel_coords[:, 1] = torch.clamp(voxel_coords[:, 1], 0, height - 1)
        voxel_coords[:, 2] = torch.clamp(voxel_coords[:, 2], 0, width - 1)
        
        # Create Gaussian kernel for smoothing
        kernel_size = int(3 * sigma + 1)  # 3-sigma rule
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Set voxel values with Gaussian smoothing
        for i in range(voxel_coords.shape[0]):
            z, y, x = voxel_coords[i]
            
            # Define neighborhood for Gaussian
            z_min = max(0, z - kernel_size // 2)
            z_max = min(depth, z + kernel_size // 2 + 1)
            y_min = max(0, y - kernel_size // 2)
            y_max = min(height, y + kernel_size // 2 + 1)
            x_min = max(0, x - kernel_size // 2)
            x_max = min(width, x + kernel_size // 2 + 1)
            
            # Apply Gaussian weights to neighborhood
            for zi in range(z_min, z_max):
                for yi in range(y_min, y_max):
                    for xi in range(x_min, x_max):
                        # Calculate Gaussian weight
                        dist_sq = (zi - z)**2 + (yi - y)**2 + (xi - x)**2
                        weight = torch.exp(-dist_sq / (2 * sigma**2))
                        
                        # Add weighted intensity (accumulate for overlapping points)
                        pred_volume[b, 0, zi, yi, xi] += intensity * weight
    
    # Normalize to prevent oversaturation from overlapping points
    max_val = pred_volume.max()
    if max_val > intensity:
        pred_volume = pred_volume / max_val * intensity
    
    return pred_volume


def generate_depth_map_from_volume(volume_tensor, projector, projection_angle=0.0, intensity_threshold=0.01):
    """
    Generate accurate depth map by ray-tracing through the 3D volume.
    
    This method computes the actual depth by finding where rays from the X-ray source
    first intersect with anatomy (voxels above threshold) in the 3D volume.
    
    Args:
        volume_tensor: Input volume tensor [1, D, H, W] or [D, H, W]
        projector: ConeBeam3DProjector instance with geometry information
        projection_angle: Specific angle for depth map generation
        intensity_threshold: Minimum voxel intensity to consider as anatomy
    
    Returns:
        depth_map: 2D depth map [H, W] with actual distances from source to first intersection
    """
    try:
        # Ensure volume has batch dimension
        if volume_tensor.ndim == 3:
            volume = volume_tensor.unsqueeze(0)  # Add batch dimension
        else:
            volume = volume_tensor
        
        # Get volume dimensions
        batch_size, depth_dim, height_dim, width_dim = volume.shape
        
        # Get the geometry from projector
        image_size = [depth_dim, height_dim, width_dim]
        proj_size = [height_dim, width_dim]  # Assume square projection
        
        # Create geometry parameters for single angle
        geo_param = Initialization_ConeBeam(
            image_size=image_size,
            num_proj=1,
            start_angle=projection_angle,
            proj_size=proj_size,
            raw_reso=0.7
        )
        
        # Build ODL geometry
        reco_space, ray_trafo, _ = build_conebeam_gemotry(geo_param)
        geometry = ray_trafo.geometry
        
        # Get source and detector positions
        source_position = geometry.src_position(projection_angle)  # Source position for this angle
        detector_positions = geometry.det_point_position(projection_angle, geometry.det_partition.points)
        
        # Convert volume to numpy for ray tracing
        volume_np = volume.squeeze(0).cpu().numpy()  # Remove batch dimension
        
        # Initialize depth map
        depth_map = np.zeros((proj_size[0], proj_size[1]), dtype=np.float32)
        
        # Physical volume bounds (from ODL geometry)
        volume_min = np.array([-geo_param.param['sx']/2, -geo_param.param['sy']/2, -geo_param.param['sz']/2])
        volume_max = np.array([geo_param.param['sx']/2, geo_param.param['sy']/2, geo_param.param['sz']/2])
        volume_spacing = (volume_max - volume_min) / np.array([width_dim, height_dim, depth_dim])
        
        # Ray trace for each detector pixel
        det_idx = 0
        for i in range(proj_size[0]):
            for j in range(proj_size[1]):
                # Get detector position for this pixel
                detector_pos = detector_positions[det_idx]
                det_idx += 1
                
                # Ray direction from source to detector
                ray_direction = detector_pos - source_position
                ray_direction = ray_direction / np.linalg.norm(ray_direction)  # Normalize
                
                # Ray tracing through volume (use fast version for better performance)
                depth = trace_ray_through_volume_fast(
                    volume_np, 
                    source_position, 
                    ray_direction,
                    volume_min, 
                    volume_max, 
                    volume_spacing,
                    intensity_threshold
                )
                
                depth_map[i, j] = depth
        
        return depth_map
        
    except Exception as e:
        print(f"Warning: Failed to generate volume-based depth map: {e}")
        raise ValueError(f"Depth generation failed for model")


def trace_ray_through_volume(volume, source_pos, ray_dir, vol_min, vol_max, vol_spacing, threshold):
    """
    Ray trace through 3D volume to find first intersection with anatomy.
    
    Uses adaptive step size for efficient ray marching through the volume.
    
    Args:
        volume: 3D numpy array [D, H, W]
        source_pos: 3D source position
        ray_dir: 3D normalized ray direction
        vol_min: Volume minimum bounds [3]
        vol_max: Volume maximum bounds [3] 
        vol_spacing: Voxel spacing [3]
        threshold: Intensity threshold for anatomy detection
    
    Returns:
        depth: Distance from source to first intersection, or max distance if no intersection
    """
    # Ray parameters
    max_distance = np.linalg.norm(vol_max - vol_min) * 1.5  # Maximum reasonable distance
    min_step = min(vol_spacing) * 0.3  # Fine sampling for accuracy
    max_step = min(vol_spacing) * 2.0  # Coarse sampling for speed
    
    t = 0.0  # Current distance along ray
    step_size = max_step  # Start with coarse sampling
    
    while t < max_distance:
        # Current position along ray
        current_pos = source_pos + t * ray_dir
        
        # Check if position is within volume bounds
        if np.any(current_pos < vol_min) or np.any(current_pos > vol_max):
            t += step_size
            continue
            
        # Convert world coordinates to voxel indices
        voxel_coords = (current_pos - vol_min) / vol_spacing
        voxel_indices = np.round(voxel_coords).astype(int)
        
        # Check bounds
        if (voxel_indices[0] >= 0 and voxel_indices[0] < volume.shape[2] and
            voxel_indices[1] >= 0 and voxel_indices[1] < volume.shape[1] and
            voxel_indices[2] >= 0 and voxel_indices[2] < volume.shape[0]):
            
            # Sample volume at this position (note: volume is [D, H, W])
            intensity = volume[voxel_indices[2], voxel_indices[1], voxel_indices[0]]
            
            # Check if we're getting close to anatomy (adaptive step sizing)
            if intensity > threshold * 0.1:  # Getting close
                if step_size > min_step:
                    step_size = min_step  # Switch to fine sampling
                    
                # Check if we hit anatomy
                if intensity > threshold:
                    return t  # Return distance to first intersection
            else:
                step_size = max_step  # Use coarse sampling in empty regions
        
        t += step_size
    
    # No intersection found - return a large depth value
    return max_distance * 0.9  # Return 90% of max distance to indicate background


def trace_ray_through_volume_fast(volume, source_pos, ray_dir, vol_min, vol_max, vol_spacing, threshold):
    """
    Fast vectorized ray tracing through volume.
    Alternative implementation for better performance with large volumes.
    """
    try:
        # Calculate entry and exit points for the volume bounding box
        t_min = (vol_min - source_pos) / (ray_dir + 1e-8)  # Add small epsilon to avoid division by zero
        t_max = (vol_max - source_pos) / (ray_dir + 1e-8)
        
        # Ensure t_min < t_max for each axis
        t_near = np.minimum(t_min, t_max)
        t_far = np.maximum(t_min, t_max)
        
        # Ray enters volume at max(t_near) and exits at min(t_far)
        t_enter = np.max(t_near)
        t_exit = np.min(t_far)
        
        # If t_enter > t_exit, ray doesn't intersect volume
        if t_enter > t_exit or t_exit < 0:
            return vol_max[0] * 2  # Return large distance
        
        # Start ray marching from entry point
        t_start = max(0, t_enter)
        step_size = min(vol_spacing) * 0.5
        
        t = t_start
        while t <= t_exit:
            # Current position along ray
            current_pos = source_pos + t * ray_dir
            
            # Convert to voxel coordinates
            voxel_coords = (current_pos - vol_min) / vol_spacing
            voxel_indices = np.clip(np.round(voxel_coords).astype(int), 
                                  [0, 0, 0], 
                                  [volume.shape[2]-1, volume.shape[1]-1, volume.shape[0]-1])
            
            # Sample volume (note: volume is [D, H, W], coordinates are [X, Y, Z])
            intensity = volume[voxel_indices[2], voxel_indices[1], voxel_indices[0]]
            
            if intensity > threshold:
                return t
            
            t += step_size
        
        # No intersection found
        return t_exit * 1.1
        
    except Exception as e:
        # Fallback to basic ray tracing
        return trace_ray_through_volume(volume, source_pos, ray_dir, vol_min, vol_max, vol_spacing, threshold)


class GCPDataset(Dataset):
    """
    Dataset for training Gaussian Center Predictor.
    
    Loads 3D volumes and generates required ground truth data:
    - projection: Generated from volume using CT projector
    - point_cloud: Extracted from volume voxels
    - volume: Original 3D volume
    - depth_map: Generated from projection geometry
    """
    
    def __init__(self, data_folder: str, split: str = 'train', train_ratio: float = 0.9, num_proj: int = 1):
        self.data_folder = data_folder
        self.split = split
        self.num_proj = num_proj
        
        # Initialize CT projector for generating projections and depth maps
        image_size = [128, 128, 128]
        proj_size = [128, 128]
        self.projector = ConeBeam3DProjector(image_size, proj_size, num_proj=num_proj)
        
        # Find all .npy volume files
        npy_files = glob.glob(os.path.join(data_folder, "*.npy"))
        model_indices = []
        for file in npy_files:
            filename = os.path.basename(file)
            if filename.replace('.npy', '').isdigit():
                model_indices.append(int(filename.replace('.npy', '')))
        model_indices.sort()
        
        # Split into train/validation
        split_idx = int(len(model_indices) * train_ratio)
        if split == 'train':
            self.model_indices = model_indices[:split_idx]
        else:  # validation
            self.model_indices = model_indices[split_idx:]
        
        print(f"Found {len(self.model_indices)} samples for {split} split")
    
    def __len__(self):
        return len(self.model_indices)
    
    def __getitem__(self, idx):
        model_id = self.model_indices[idx]
        
        # Load 3D volume
        model_path = os.path.join(self.data_folder, f"{model_id}.npy")
        volume = np.load(model_path).astype(np.float32)
        
        # Convert to torch tensor and add batch dimension
        volume_tensor = torch.from_numpy(volume).float()
        if volume_tensor.ndim == 3:
            volume_tensor = volume_tensor.unsqueeze(0)  # Add batch dimension [1, D, H, W]
        
        # Generate projection from volume
        with torch.no_grad():
            # Move to GPU temporarily for projection
            volume_gpu = volume_tensor.cuda()
            projection_tensor = self.projector.forward_project(volume_gpu)  # [1, num_proj, H, W]
            projection_tensor = projection_tensor.cpu()
            volume_tensor = volume_tensor.cpu()
        
        # Take first projection if multiple
        projection = projection_tensor[0, 0].numpy()  # [H, W]
        
        # Normalize projection to [0, 1]
        if projection.max() > projection.min():
            projection = (projection - projection.min()) / (projection.max() - projection.min())
        else:
            projection = np.zeros_like(projection)
        
        # Extract point cloud from volume (voxel positions with intensity > threshold)
        threshold = 0.1
        points_indices = np.where(volume > threshold)
        if len(points_indices[0]) > 0:
            points = np.column_stack(points_indices).astype(np.float32)
        else:
            raise ValueError(f"No points found in volume {model_id} above threshold {threshold}")
        
        # Generate accurate depth map using volume ray-tracing
        try:
            depth = generate_depth_map_from_volume(
                volume_tensor.cpu(), 
                self.projector, 
                projection_angle=0.0,  # Use first projection angle
                intensity_threshold=0.05  # Threshold for detecting anatomy
            )
            
            # Debug information for depth map quality
            meaningful_pixels = np.sum(depth < depth.max() * 0.9)
            depth_range = depth.max() - depth.min()
            
            if meaningful_pixels < depth.size * 0.1:  # Less than 10% meaningful pixels
                print(f"Warning: Model {model_id} depth map has only {meaningful_pixels}/{depth.size} meaningful pixels")
            
            if depth_range < 1.0:  # Very small depth range
                print(f"Warning: Model {model_id} has very small depth range: {depth_range:.3f}")
                
        except Exception as e:
            print(f"Warning: Volume-based depth generation failed for model {model_id}: {e}")
            raise ValueError(f"Depth generation failed for model {model_id}")
        
        # Create depth mask (areas with valid depth)
        depth_mask = (depth > 0).astype(np.float32)
        
        # Add channel dimensions
        projection = projection[np.newaxis, ...]  # [1, H, W]
        volume = volume[np.newaxis, ...]  # [1, D, H, W]
        
        return {
            'projection': torch.from_numpy(projection),
            'points': torch.from_numpy(points),
            'volume': torch.from_numpy(volume),
            'depth': torch.from_numpy(depth),
            'depth_mask': torch.from_numpy(depth_mask),
            'model_id': str(model_id)
        }


def train_gcp_model(args):
    """
    Train the Gaussian Center Predictor model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if dataset folder exists
    if not os.path.exists(args.data_folder):
        print(f"Error: Dataset folder '{args.data_folder}' not found!")
        print("Please provide a valid path to the folder containing .npy volume files.")
        return
    
    # Check if there are any .npy files in the folder
    npy_files = glob.glob(os.path.join(args.data_folder, "*.npy"))
    if not npy_files:
        print(f"Error: No .npy files found in '{args.data_folder}'!")
        print("Please ensure the folder contains 3D volume files in .npy format.")
        return
    
    print(f"Found {len(npy_files)} volume files in dataset folder")
    
    # Create datasets
    train_dataset = GCPDataset(
        args.data_folder, 
        split='train', 
        train_ratio=args.train_ratio,
        num_proj=args.num_proj
    )
    val_dataset = GCPDataset(
        args.data_folder, 
        split='val', 
        train_ratio=args.train_ratio,
        num_proj=args.num_proj
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = create_gcp_network(
        downsampling_factor=args.downsampling_factor
    ).to(device)
    
    # Create loss function and optimizer
    criterion = GCPLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    
    for epoch in range(start_epoch, args.num_epochs):
        # Training phase
        model.train()
        epoch_train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            projection = batch['projection'].to(device)
            gt_points = batch['points'].to(device)
            gt_volume = batch['volume'].to(device)
            gt_depth = batch['depth'].to(device)
            depth_mask = batch['depth_mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get model predictions
            pred_centers, pred_depths, _ = model.predict_gaussian_centers(projection)
            
            # Create predicted volume from centers using voxelization
            pred_volume = voxelize_point_cloud(pred_centers, gt_volume.shape, device)
            
            # Calculate current iteration for dynamic weighting
            iteration = epoch * len(train_loader) + batch_idx + 1
            
            # Compute loss
            loss, loss_dict = criterion(
                pred_centers, pred_volume, pred_depths,
                gt_points, gt_volume, gt_depth,
                depth_mask, iteration
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
            
            # Log progress
            print(f'Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx}/{len(train_loader)}, 'f'Loss: {loss.item():.6f}')
            print(f'Chamfer: {loss_dict["chamfer_loss"]:.6f}, clDice: {loss_dict["cldice_loss"]:.6f}, Depth: {loss_dict["depth_loss"]:.6f}')
                
        # Validation phase
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                projection = batch['projection'].to(device)
                gt_points = batch['points'].to(device)
                gt_volume = batch['volume'].to(device)
                gt_depth = batch['depth'].to(device)
                depth_mask = batch['depth_mask'].to(device)
                
                # Forward pass
                pred_centers, pred_depths, _ = model.predict_gaussian_centers(projection)
                pred_volume = voxelize_point_cloud(pred_centers, gt_volume.shape, device)
                
                # Compute loss
                loss, _ = criterion(
                    pred_centers, pred_volume, pred_depths,
                    gt_points, gt_volume, gt_depth,
                    depth_mask, iteration=20000  # Use fixed iteration for validation
                )
                
                epoch_val_losses.append(loss.item())
        
        # Calculate average losses
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{args.num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(args.save_dir, 'best_gcp_model.pth'))
            print(f'New best model saved with validation loss: {best_val_loss:.6f}')
        
        # Step scheduler
        scheduler.step()
        
    # Save final model and training history
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_gcp_model.pth'))
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved in: {args.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Gaussian Center Predictor")
    
    # Data arguments
    parser.add_argument('--data-folder', type=str, default='./ToyData',
                       help='Path to dataset folder containing .npy volume files')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Ratio of data to use for training (rest for validation)')
    parser.add_argument('--num-proj', type=int, default=1,
                       help='Number of projections to generate from each volume')
    
    # Model arguments
    parser.add_argument('--downsampling-factor', type=int, default=2,
                       help='Downsampling factor for output resolution')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Scheduler arguments
    parser.add_argument('--lr-step-size', type=int, default=30,
                       help='Learning rate scheduler step size')
    parser.add_argument('--lr-gamma', type=float, default=0.5,
                       help='Learning rate scheduler gamma')
    
    # Logging and saving arguments
    parser.add_argument('--save-dir', type=str, default='./gcp_checkpoints',
                       help='Directory to save models')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Start training
    train_gcp_model(args)


if __name__ == "__main__":
    main()