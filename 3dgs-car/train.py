import os
import sys
import copy
import time
import glob
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss

from arguments_init import *
from ct_geometry_projector import ConeBeam3DProjector
from gaussian_model_anisotropic import GaussianModelAnisotropic
from gaussian_center_predictor import create_gcp_network
from gaussian_losses import CombinedGaussianLoss, compute_psnr

cudnn.benchmark = True

def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid

def evaluate_gaussian(dataset_folder, num_proj, save_dir, opt, args):
    image_size = [128] * 3
    proj_size = [128] * 3
    ct_projector_train = ConeBeam3DProjector(image_size, proj_size, num_proj)
    ct_projector_new = ConeBeam3DProjector(image_size, proj_size, num_proj, start_angle=5)
    os.makedirs(save_dir, exist_ok=True)

    # Determine the range of models to train on
    if args.model_range is not None:
        start_model, end_model = args.model_range
        model_indices = list(range(start_model, end_model + 1))  # inclusive range
        print(f"Training on models {start_model} to {end_model} (range specified)")
    else:
        # Find all .npy files in the dataset folder
        npy_files = glob.glob(os.path.join(dataset_folder, "*.npy"))
        model_indices = []
        for file in npy_files:
            filename = os.path.basename(file)
            if filename.replace('.npy', '').isdigit():
                model_indices.append(int(filename.replace('.npy', '')))
        model_indices.sort()
        print(f"Training on all {len(model_indices)} models found in dataset folder")
    
    if not model_indices:
        print("No valid model files found!")
        return

    for i, model_id in enumerate(model_indices):
        print(f"Start to evaluate model {model_id} ({i+1}/{len(model_indices)}) with {num_proj} views")
        best_psnr, patient, best_iter = 0, 0, 0

        # Load GT volume from individual numpy file
        model_path = os.path.join(dataset_folder, f"{model_id}.npy")
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found, skipping...")
            continue
            
        gt_volume = torch.from_numpy(np.load(model_path)).float().cuda()
        if gt_volume.ndim == 3:
            gt_volume = gt_volume.unsqueeze(0)  # Add batch dimension if needed

        # prepare gaussian model
        opt.density_lr = args.density_lr
        opt.sigma_lr = args.sigma_lr
        gaussians = GaussianModelAnisotropic()

        input_projs = ct_projector_train.forward_project(gt_volume)   # [1, num_proj, x, y]

        # Initialize Gaussians based on the chosen method
        if args.init_method == 'gcp' and args.gcp_model_path and os.path.exists(args.gcp_model_path):
            print(f"Initializing Gaussians using GCP model: {args.gcp_model_path}")
            
            # Load pre-trained GCP model
            gcp_model = create_gcp_network(downsampling_factor=2)
            checkpoint = torch.load(args.gcp_model_path, map_location='cuda')
            if 'model_state_dict' in checkpoint:
                gcp_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                gcp_model.load_state_dict(checkpoint)
            gcp_model.cuda()
            gcp_model.eval()
            
            # Use first projection for GCP initialization
            first_proj = input_projs[:, 0:1]  # [1, 1, H, W]
            
            with torch.no_grad():
                gcp_centers, _, _ = gcp_model.predict_gaussian_centers(first_proj)
                # gcp_centers: [1, N, 3], we want [N, 3]
                gcp_centers = gcp_centers.squeeze(0)  # [N, 3]
            
            # Limit number of centers if specified
            if args.num_init_gaussian > 0 and gcp_centers.shape[0] > args.num_init_gaussian:
                # Randomly sample or take first N centers
                indices = torch.randperm(gcp_centers.shape[0])[:args.num_init_gaussian]
                gcp_centers = gcp_centers[indices]
            
            # Initialize from GCP predictions
            gaussians.create_from_gcp(
                gcp_centers, 
                ini_density=0.04, 
                ini_sigma=0.01, 
                spatial_lr_scale=1,
                volume_size=image_size
            )
            
        else:
            print("Initializing Gaussians using FBP method")
            # Original FBP initialization
            fbp_recon = ct_projector_train.backward_project(input_projs)
            gaussians.create_from_fbp(
                fbp_recon, 
                air_threshold=0.05, 
                ini_density=0.04, 
                ini_sigma=0.01, 
                spatial_lr_scale=1, 
                num_samples=args.num_init_gaussian
            )
        
        gaussians.training_setup(opt)

        max_iter = args.max_iter
        starttime = time.time()
        for iteration in range(max_iter):
            # Forward pass
            gaussians.update_learning_rate(iteration)
            grid = create_grid_3d(*image_size)
            grid = grid.cuda()
            grid = grid.unsqueeze(0).repeat(input_projs.shape[0], 1, 1, 1, 1)
            train_output = gaussians.grid_sample(grid, expand=[5, 15, 15])
            del grid
            torch.cuda.empty_cache()

            # Loss - use combined loss function
            train_projs = ct_projector_train.forward_project(train_output.transpose(1,4).squeeze(1))
            
            if args.use_combined_loss:
                # Use the combined loss from the paper: L_G = Σ_i (α * L_L2 + (1-α) * L_clL2)
                loss_fn = CombinedGaussianLoss(alpha=args.loss_alpha)
                loss, loss_dict = loss_fn(train_projs, input_projs)
                
                # Log detailed loss components
                if iteration == 0 or (iteration + 1) % 500 == 0:
                    print(f"  L2 Loss: {loss_dict['l2_loss']:.6f}, "
                          f"Centerline Loss: {loss_dict['centerline_loss']:.6f}")
            else:
                # Original MSE loss
                loss = mse_loss(train_projs, input_projs)
            
            loss.backward()
            
            # Compute PSNR for monitoring
            train_psnr = compute_psnr(train_projs, input_projs)

            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 1.5)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration == 0 or (iteration + 1) % 100 == 0:
                if train_psnr > best_psnr:
                    best_psnr = train_psnr
                    patient = 0
                    saved_model = copy.deepcopy(gaussians.state_dict())
                    best_iter = iteration
                    print("best_psnr: ", best_psnr)
                else:
                    patient += 1
                    if patient > 20:
                        print("Early stopping at iteration: ", best_iter, "best_psnr: ", best_psnr)
                        break

        endtime = time.time()
        print(f"Training time: {endtime - starttime}")
        gaussians.load_state_dict(saved_model)

        # Testing
        with torch.no_grad():
            grid = create_grid_3d(*image_size)
            grid = grid.cuda()
            # train_data[0] grid: [batchsize, z, x, y, 3]
            grid = grid.unsqueeze(0).repeat(input_projs.shape[0], 1, 1, 1, 1)
            train_output = gaussians.grid_sample(grid, expand=[15, 15, 15])
            del grid
            torch.cuda.empty_cache()

        # evaluate voxel result
        fbp_recon = train_output.transpose(1,4).squeeze(1).detach()

        # save fbp_recon in numpy format
        fbp_recon_saved = fbp_recon.squeeze(0).detach().cpu().numpy()
        np.save(os.path.join(save_dir, "recon_" + str(model_id) + "_views_" + str(num_proj) + '.npy'), fbp_recon_saved)
        
        # Also save ground truth volume in numpy format for comparison
        gt_volume_saved = gt_volume.squeeze(0).detach().cpu().numpy()
        np.save(os.path.join(save_dir, "gt_volume_" + str(model_id) + ".npy"), gt_volume_saved)

        #generate new projs
        new_projs_tr = ct_projector_new.forward_project(fbp_recon)
        new_projs_gt = ct_projector_new.forward_project(gt_volume)

        # save new_projs in numpy format
        new_projs_tr_saved = new_projs_tr.detach().cpu().numpy()
        new_projs_gt_saved = new_projs_gt.detach().cpu().numpy()
        
        np.save(os.path.join(save_dir, "recon_" + str(model_id) + "_views_" + str(num_proj) + '_new_projs_train.npy'), new_projs_tr_saved)
        np.save(os.path.join(save_dir, "recon_" + str(model_id) + "_views_" + str(num_proj) + '_new_projs_label.npy'), new_projs_gt_saved)


if __name__ == "__main__":
    cpu_num = 10
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument('--mydensity_lr', type=float, default=1e-2)
    parser.add_argument('--mysigma_lr', type=float, default=1e-2)
    
    parser.add_argument('--max_iter', type=int, default=20000)
    parser.add_argument('--num_init_gaussian', type=int, default=10000)
    parser.add_argument('--num_proj', type=int, default=2)
    parser.add_argument('--model_range', nargs=2, type=int, default=None, 
                       help='Range of models to train [start, end] (inclusive). Example: --model_range 900 1000')
    parser.add_argument('--dataset_folder', type=str, default="/kaggle/working/ToyData/",
                       help='Path to folder containing individual model .npy files')
    
    # GCP initialization arguments
    parser.add_argument('--init_method', type=str, choices=['fbp', 'gcp'], default='fbp',
                       help='Initialization method: fbp (Filtered Back Projection) or gcp (Gaussian Center Predictor)')
    parser.add_argument('--gcp_model_path', type=str, default=None,
                       help='Path to pre-trained GCP model for initialization')
    
    # Loss function arguments
    parser.add_argument('--use_combined_loss', action='store_true',
                       help='Use the combined loss function from the paper (L2 + centerline loss)')
    parser.add_argument('--loss_alpha', type=float, default=0.5,
                       help='Weight for L2 loss in combined loss function (1-alpha for centerline loss)')
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Set dataset path from arguments
    dataset_path = args.dataset_folder
    save_dir = "/kaggle/working/results/"

    evaluate_gaussian(dataset_path, args.num_proj, save_dir, op.extract(args), args)