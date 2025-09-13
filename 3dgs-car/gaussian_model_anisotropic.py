import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from simple_knn._C import distCUDA2

from utils.general_utils import build_scaling_rotation, inverse_sigmoid, get_expon_lr_func


class GaussianModelAnisotropic:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance
        self._scaling_activation = torch.sigmoid
        self.scaling_inverse_activation = inverse_sigmoid

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.sigmoid
        self.inverse_density_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._density = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    @property
    def get_scaling(self):
        #return self._scaling_activation(self._sigma)
        return self._scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_gaussians_num(self):
        return self._xyz.shape[0]
    
    @property
    def get_density(self):
        return self.density_activation(self._density)
    
    @property
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)
    
    def create_from_fbp(self, fbp_image, air_threshold=0.05, ini_density=0.04, ini_sigma=0.01, spatial_lr_scale=1, num_samples=150000):
        self.spatial_lr_scale = spatial_lr_scale

        # convert shape of fbp_image from [bs,z,x,y] to [bs,z,x,y,1]
        fbp_image = fbp_image.unsqueeze(-1)

        # fbp recon:[bs,z,x,y,1]
        bs, D, H, W, _ = fbp_image.shape

        fbp_image = fbp_image.permute(0, 4, 1, 2, 3)  # [bs,1,z,x,y]
        fbp_image = F.interpolate(fbp_image, size=(H, H, W), mode='trilinear', align_corners=False)

        # Calculate gradients in 3 directions for each voxel
        grad_x = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 1:-1, 1:-1, 2:])
        grad_y = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 1:-1, 2:, 1:-1])
        grad_z = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 2:, 1:-1, 1:-1])

        # Pad at the beginning and end of each dimension
        grad_x_padded = F.pad(grad_x, (1, 1, 1, 1, 1, 1), "constant", 0)
        grad_y_padded = F.pad(grad_y, (1, 1, 1, 1, 1, 1), "constant", 0)
        grad_z_padded = F.pad(grad_z, (1, 1, 1, 1, 1, 1), "constant", 0)

        # Calculate the norm of gradients for each voxel [bs,1,z,X,y]
        grad_norm = torch.sqrt(grad_x_padded ** 2 + grad_y_padded ** 2 + grad_z_padded ** 2)

        # Sort by gradient magnitude and select indices of num_samples voxels with largest gradients
        grad_norm = grad_norm.reshape(-1)
        _, indices = torch.topk(grad_norm, num_samples)

        # Extract 3D coordinates corresponding to these indices
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(H), torch.arange(W)), dim=-1).reshape(-1,3).cuda()

        # Actually, the current implementation should be slightly misaligned because boundary voxels are removed
        sampled_coords = coords[indices]

        # Create a 3D grid to count the number of sampled points around each point
        grid = torch.zeros((H, H, W), dtype=torch.int32, device="cuda")

        # Increase the count in the grid at the location of each sampled point
        indices_3d = sampled_coords.long()
        grid[indices_3d[:, 0], indices_3d[:, 1], indices_3d[:, 2]] += 1

        # Set densities proportional to FBP image values
        fbp_image[fbp_image < air_threshold] = 0
        densities = ini_density * fbp_image.reshape(-1)[indices]
        sampled_coords = sampled_coords.float()

        # Normalize to [0, 1]
        sampled_coords = sampled_coords / torch.tensor([H, H, W], dtype=torch.float, device="cuda")
        
        # For each sampled_coord, the more sampled_coords around it, the smaller its sigma; otherwise sigma becomes larger; densities all use consistent values
        densities = inverse_sigmoid(densities).unsqueeze(1)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(sampled_coords.cpu())).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((num_samples, 4), device="cuda")
        rots[:, 0] = 1
        self._xyz = nn.Parameter(sampled_coords.requires_grad_(True))
        self._density = nn.Parameter(densities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._density], 'lr': training_args.density_lr, "name": "density"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_densities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "density": new_densities,
        "scaling": new_scaling,
        "rotation": new_rotation
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling,dim=1).values > self.percent_dense * scene_extent)
        print("=================================================")
        print("Spliting {} points".format(selected_pts_mask.sum()))

        if self.get_gaussians_num + selected_pts_mask.sum() < 12000:
            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            new_xyz = samples + self.get_xyz[selected_pts_mask].repeat(N, 1)

            new_densities = self._density[selected_pts_mask].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
            new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

            self.densification_postfix(new_xyz, new_densities, new_scaling, new_rotation)

            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        print("=================================================")
        print("Cloning {} points".format(selected_pts_mask.sum()))

        new_xyz = self._xyz[selected_pts_mask]

        new_density = self._density[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_density, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_density, sigma_extent, depmapX=None, depmapY=None):
        grads = torch.norm(self._xyz.grad, dim=-1, keepdim=True)

        self.densify_and_clone(grads, max_grad, sigma_extent)
        self.densify_and_split(grads, max_grad, sigma_extent)

        prune_mask = (self.get_density < min_density).squeeze()

        print("Pruning {} points".format(prune_mask.sum()))
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def grid_sample(self, grid, expand):
        # grid: [batchsize, z, x, y, 3]
        # expand dimensions for broadcasting
        grid_expanded = grid.unsqueeze(-2)  # [batchsize, z, x, y, 1, 3]

        # compute density for each point in the grid
        density_grid = self.compute_density(self._xyz, grid_expanded, self.get_density, self.get_covariance, expand)
  
        return density_grid#.unsqueeze(-1)  # [batchsize, z, x, y, num_gaussians, 1]
    
    def compute_density(self, gaussian_centers, grid_point, density, covariance, expand=[5,15,15]):
        # grid_point: [1, z, x, y, 1, 3]
        z,x,y = grid_point.shape[1:4]
        num_gaussians = gaussian_centers.shape[-2]
        # initialize density_grid outside the loop
        density_grid = torch.zeros(1, z, x, y, 1, device='cuda')
        expanded_grid_point = grid_point.expand(num_gaussians,z,x,y,1,3)

        mean_zxy = gaussian_centers * torch.tensor([z, x, y]).cuda() # [num_gaussian, 3]
        mean_z, mean_x, mean_y = mean_zxy[:,0], mean_zxy[:,1], mean_zxy[:,2]
        # When calculating distance, index num_gaussians_in_batch small patches instead of entire large patch, calculate distance with each gaussian_center separately
        # PyTorch only supports indexing patches of uniform size, so use fixed size
        z_indices = torch.clamp((mean_z.unsqueeze(-1)-expand[0]/2).int() + torch.arange(0, expand[0], device='cuda'), 0, z-1) #[num_gaussian_in_patch, expand[0]]
        x_indices = torch.clamp((mean_x.unsqueeze(-1)-expand[1]/2).int() + torch.arange(0, expand[1], device='cuda'), 0, x-1) #[num_gaussian_in_patch, expand[1]]
        y_indices = torch.clamp((mean_y.unsqueeze(-1)-expand[2]/2).int() + torch.arange(0, expand[2], device='cuda'), 0, y-1) #[num_gaussian_in_patch, expand[2]]

        grid_indices = torch.arange(num_gaussians, device='cuda').view(-1, 1, 1, 1)
        z_indices = z_indices.view(num_gaussians, -1, 1, 1) # [num_gaussians_in_batch, expand[0], 1, 1]
        x_indices = x_indices.view(num_gaussians, 1, -1, 1) # [num_gaussians_in_batch, 1, expand[1], 1]
        y_indices = y_indices.view(num_gaussians, 1, 1, -1) # [num_gaussians_in_batch, 1, 1, expand[2]]
        patches = expanded_grid_point[grid_indices, z_indices, x_indices, y_indices, :, :] # [num_gaussians_in_batch, expand[0], expand[1], expand[2], 1, 3]
        regularization_term = 1e-6 * torch.eye(3, device='cuda')
        regularized_covariance = covariance + regularization_term
        density_patch = (density.view(-1, 1, 1, 1, 1) * torch.exp(-0.5 * torch.matmul(torch.matmul((patches - gaussian_centers.view(num_gaussians, 1,1,1,1, 3)).unsqueeze(-2), torch.inverse(regularized_covariance.view(num_gaussians, 1,1,1,1, 3, 3))), (patches - gaussian_centers.view(num_gaussians, 1,1,1,1, 3)).unsqueeze(-1)).squeeze(-1).squeeze(-1))) # [num_gaussian_in_patch, expand[0], expand[1], expand[2], 1]
        # Prepare indices for adding the patch back to the density_grid
        indices = ((z_indices * x + x_indices) * y + y_indices).view(-1)
        # Add the density patch back to the density_grid
        density_grid = density_grid.view(-1) # [1*z*x*y*1]
        density_patch = density_patch.view(-1) # [num_gaussians_in_batch*expand[0]*expand[1]*expand[2]*1]
        
        density_grid.scatter_add_(0, indices, density_patch)
        # Reshape density_grid back to its original shape
        density_grid = density_grid.view(1, z, x, y, 1)

        return density_grid
    
    def state_dict(self):
        return {
            '_xyz': self._xyz,
            '_density': self._density,
            '_scaling': self._scaling,
            '_rotation': self._rotation,
            # Add other parameters that need to be saved
        }

    def load_state_dict(self, state_dict):
        self._xyz = state_dict['_xyz']
        self._density = state_dict['_density']
        self._scaling = state_dict['_scaling']
        self._rotation = state_dict['_rotation']
        # Load other parameters