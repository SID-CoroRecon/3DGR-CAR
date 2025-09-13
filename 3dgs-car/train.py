import os
import sys
import copy
import time
import torch
import numpy as np
import nibabel as nib
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss

from arguments_init import *
from ct_geometry_projector import ConeBeam3DProjector
from gaussian_model_anisotropic import GaussianModelAnisotropic

cudnn.benchmark = True

def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid

def evaluate_gaussian_fbp(dataset_path, num_proj, save_dir, opt, args):
    image_size = [128] * 3
    proj_size = [128] * 3
    ct_projector_train = ConeBeam3DProjector(image_size, proj_size, num_proj)
    ct_projector_new = ConeBeam3DProjector(image_size, proj_size, num_proj, start_angle=5)

    dataset = torch.load(dataset_path)['volume'].cuda()
    models_no = dataset.shape[0]

    for model in range(models_no):
        print("Start to evaluate model " + str(model) + " with " + str(num_proj) + " views")
        best_psnr, patient, best_iter = 0, 0, 0

        # prepare gaussian model
        opt.density_lr = args.density_lr
        opt.sigma_lr = args.sigma_lr
        gaussians = GaussianModelAnisotropic()

        # volume data CAS
        gt_volume = dataset[model].cuda()
        input_projs = ct_projector_train.forward_project(gt_volume)   # [1, num_proj, x, y]

        # fbp initial gaussian model
        fbp_recon = ct_projector_train.backward_project(input_projs)
        gaussians.create_from_fbp(fbp_recon, air_threshold=0.05, ini_density=0.04, ini_sigma=0.01, spatial_lr_scale=1, num_samples=args.num_init_gaussian)
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

            # Loss
            train_projs = ct_projector_train.forward_project(train_output.transpose(1,4).squeeze(1))
            loss = mse_loss(train_projs, input_projs)
            loss.backward()
            train_psnr = -10 * torch.log10(loss).item()

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
                    if patient > 6:
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

        # save fbp_recon
        fbp_recon_saved = fbp_recon.squeeze(0).detach().cpu().numpy()
        fbp_recon_saved = nib.Nifti1Image(fbp_recon_saved, np.eye(4))
        nib.save(fbp_recon_saved, os.path.join(save_dir, "recon_" + str(model) + "_views_" + str(num_proj) + '.nii.gz'))

        #generate new projs
        new_projs_tr = ct_projector_new.forward_project(fbp_recon)
        new_projs_gt = ct_projector_new.forward_project(gt_volume)

        # save new_projs to torch pt file
        torch.save(new_projs_tr, os.path.join(save_dir, "recon_" + str(model) + "_views_" + str(num_proj) + '_new_projs_train.pt'))
        torch.save(new_projs_gt, os.path.join(save_dir, "recon_" + str(model) + "_views_" + str(num_proj) + '_new_projs_label.pt'))


if __name__ == "__main__":
    cpu_num = 10
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    dataset_path = r"/kaggle/working/ToyData/Normal_1.mha_volume.pt"

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
    
    parser.add_argument('--max_iter', type=int, default=8000)
    parser.add_argument('--num_init_gaussian', type=int, default=10000)
    parser.add_argument('--num_proj', type=int, default=2)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    save_dir = "/kaggle/working/"

    evaluate_gaussian_fbp(dataset_path, args.num_proj, save_dir, op.extract(args), args)