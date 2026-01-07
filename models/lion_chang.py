# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
from models.vae_adain import Model as VAE
from models.latent_points_ada_localprior import PVCNN2Prior as LocalPrior
from utils.diffusion_pvd import DiffusionDiscretized
from utils.vis_helper import plot_points
from utils.model_helper import import_model
from diffusers import DDPMScheduler
import torch
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import logging
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist, chamfer_3DFunction, SoftChamferDistance
from third_party.PyTorchEMD.emd import earth_mover_distance

def apply_repulsion_loss(x, k=10, h=0.07):
    """
    Penalize overly clustered points to reduce outliers.
    x: (B, N, 3)
    """
    B, N, _ = x.shape
    with torch.no_grad():
        dist = torch.cdist(x, x)  # [B, N, N]
        knn_idx = dist.topk(k=k+1, largest=False).indices[:, :, 1:]  # exclude self

    neighbors = torch.gather(
        x.unsqueeze(2).expand(-1, -1, k, -1),
        1,
        knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    )
    diff = x.unsqueeze(2) - neighbors
    dist_sq = (diff ** 2).sum(-1)
    loss = torch.exp(-dist_sq / (h ** 2)).mean()
    return loss


def apply_laplacian_smoothness(x, k=16):
    """
    Laplacian smoothing loss: penalize deviation from local neighborhood mean.
    x: (B, N, 3)
    """
    B, N, _ = x.shape
    with torch.no_grad():
        idx = torch.cdist(x, x).topk(k=k+1, largest=False).indices[:, :, 1:]
    neighbors = torch.gather(
        x.unsqueeze(2).expand(-1, -1, k, -1),
        1,
        idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    )
    laplacian = x - neighbors.mean(dim=2)
    return (laplacian ** 2).mean()

def apply_patchwise_chamfer(x, y, k=8):
    """
    Average Chamfer distance over local patches.
    x, y: (B, N, 3)
    chamfer_fn: returns dist1, dist2
    """
    B, N, _ = x.shape
    with torch.no_grad():
        idx_x = torch.cdist(x, x).topk(k=k+1, largest=False).indices[:, :, 1:]
        idx_y = torch.cdist(y, y).topk(k=k+1, largest=False).indices[:, :, 1:]

    # (B*N, k, 3)
    patch_x = torch.gather(
        x.unsqueeze(2).expand(-1, -1, k, -1), 1,
        idx_x.unsqueeze(-1).expand(-1, -1, -1, 3)
    ).reshape(B * N, k, 3)

    patch_y = torch.gather(
        y.unsqueeze(2).expand(-1, -1, k, -1), 1,
        idx_y.unsqueeze(-1).expand(-1, -1, -1, 3)
    ).reshape(B * N, k, 3)

    dist1, dist2, _, _ = chamfer_3DFunction.apply(patch_x, patch_y)
    return torch.mean(dist1) + torch.mean(dist2)

import torch.nn.functional as F

def density_loss(x, y, k=16):
    def local_density(p):
        dist = torch.cdist(p, p)
        knn = dist.topk(k=k+1, largest=False).values[:, :, 1:]
        return knn.mean(dim=-1)

    d_x = local_density(x)
    d_y = local_density(y)
    return F.mse_loss(d_x, d_y)

def patchwise_chamfer(x, y, k=16):
    # x, y: (B, N, 3)
    # Divide x and y into patches (e.g., via KMeans or voxel grid)
    patches_x = get_patches(x, k)
    patches_y = get_patches(y, k)
    loss = 0
    for px, py in zip(patches_x, patches_y):
        dist1, dist2, _, _ = chamfer_3DFunction.apply(px, py)
        loss += torch.mean(dist1) + torch.mean(dist2)
    return loss / len(patches_x)

def get_patches(x: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    x: (B, N, 3) input point cloud
    k: number of neighbors
    return: (B, N, k, 3) local patches centered at each point
    Each patch is represented as relative coordinates: neighbor - center
    """
    B, N, _ = x.shape
    # (B, N, N) pairwise distance
    dist = torch.cdist(x, x)
    knn_idx = dist.topk(k=k + 1, largest=False).indices[:, :, 1:]  # exclude self

    # (B, N, k, 3)
    knn_points = torch.gather(
        x.unsqueeze(2).expand(-1, -1, k, -1),
        1,
        knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    )

    # relative coordinates: neighbors - center
    center = x.unsqueeze(2)  # (B, N, 1, 3)
    patch = knn_points - center  # (B, N, k, 3)
    return patch

def visualize_single_timestep_gradient(grad_tensor, timestep=None, save_path=None):
    """
    grad_tensor: torch.Tensor of shape (1, 8192, 1) or (8192,)
    timestep: optional, for plot title
    """
    grad_values = grad_tensor.squeeze().detach().cpu().numpy()  # (8192,)

    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(grad_values)), grad_values, linewidth=0.8)
    plt.xlabel("Point Index")
    plt.ylabel("Gradient Magnitude")
    title = f"Gradient Magnitude per Point at Timestep {timestep}" if timestep is not None else "Gradient Magnitude per Point"
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Saved single timestep gradient plot to {save_path}")
    else:
        plt.show()
        
def percentile_clip(grad, lower=1, upper=99):
    grad_flat = grad.view(-1)
    lower_val = torch.quantile(grad_flat, lower / 100.0)
    upper_val = torch.quantile(grad_flat, upper / 100.0)

    grad_clipped = torch.where(grad < lower_val, lower_val, grad)
    grad_clipped = torch.where(grad_clipped > upper_val, upper_val, grad_clipped)

    return grad_clipped.view_as(grad)

def pointcloud_to_voxel(points, res=64, padding=0.05):
    """
    Convert point cloud (B, N, 3) to occupancy voxel grid (B, res, res, res)
    Args:
        points: torch.Tensor, (B, N, 3), values in [-1, 1]
        res: int, voxel resolution (e.g., 64)
    Returns:
        occupancy: torch.Tensor, (B, res, res, res)
    """
    B, N, _ = points.shape
    voxels = torch.zeros(B, res, res, res, device=points.device)

    # Normalize to [0, res)
    coords = (points + 1.0) / 2.0 * (res - 1)  # (B, N, 3)
    coords = coords.long().clamp(0, res - 1)

    for b in range(B):
        x, y, z = coords[b].T
        voxels[b, x, y, z] = 1.0  # Mark as occupied

    return voxels

def normalize_pointcloud(pc, ref=None, scale_range=[-1, 1]):
    """
    Normalize point cloud to a fixed scale range (default [-1, 1]).

    Args:
        pc: (B, N, 3) tensor to normalize
        ref: (B, N, 3) reference point cloud to compute normalization parameters.
             If None, use `pc` itself.
        scale_range: target range to scale points into

    Returns:
        Normalized point cloud with same shape as `pc`
    """
    base = pc if ref is None else ref  # 기준이 되는 reference
    min_xyz = base.amin(dim=1, keepdim=True)  # (B, 1, 3)
    max_xyz = base.amax(dim=1, keepdim=True)  # (B, 1, 3)
    center = (max_xyz + min_xyz) / 2           # 중심 정렬
    scale = (max_xyz - min_xyz).amax(dim=2, keepdim=True) / 2  # 최대 축 기준 스케일

    pc_normalized = (pc - center) / scale
    scale_factor = (scale_range[1] - scale_range[0]) / 2
    pc_scaled = pc_normalized * scale_factor
    return pc_scaled.clamp(scale_range[0], scale_range[1])

def voxel_bce_loss(pred_points: torch.Tensor,
                   gt_points: torch.Tensor,
                   res: int = 64) -> torch.Tensor:
    """
    pred_points, gt_points: (B, N, 3), value range [-1, 1]
    res: voxel grid resolution (e.g., 64 or 128)
    returns: scalar BCE loss between voxelized pred and gt
    """

    # Normalize to [-1, 1] if not already
    gt_points_norm = normalize_pointcloud(gt_points)
    pred_points_norm = normalize_pointcloud(pred_points, ref=gt_points)

    # Convert to voxel grids
    pred_voxel = pointcloud_to_voxel(pred_points_norm, res=res)  # (B, res, res, res)
    gt_voxel = pointcloud_to_voxel(gt_points_norm, res=res)

    # Apply BCE loss
    loss = F.binary_cross_entropy(pred_voxel, gt_voxel)
    return loss

def local_smoothness_loss(x, k=8):
    with torch.no_grad():
        dist = torch.cdist(x, x)
        knn_idx = dist.topk(k=k+1, largest=False).indices[:, :, 1:]
        neighbors = torch.gather(
            x.unsqueeze(2).expand(-1, -1, k, -1),
            1,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
        )
    diff = x.unsqueeze(2) - neighbors
    return (diff**2).mean()

def curvature_loss(x, y, k=64):
    # x, y: (B, N, 3)
    # get knn neighbors for each point
    knn_x = get_knn(x, k)  # (B, N, k, 3)
    knn_y = get_knn(y, k)

    diff_x = knn_x - x.unsqueeze(2)
    diff_y = knn_y - y.unsqueeze(2)

    curvature_x = diff_x.std(dim=2)
    curvature_y = diff_y.std(dim=2)

    return F.mse_loss(curvature_x, curvature_y)


def get_knn(x: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    x: (B, N, 3) point cloud tensor
    k: number of nearest neighbors
    return: (B, N, k, 3) tensor of k-nearest neighbors for each point
    """
    B, N, _ = x.shape
    dist = torch.cdist(x, x, p=2)  # (B, N, N)
    knn_idx = dist.topk(k=k + 1, largest=False).indices[:, :, 1:]  # exclude self, (B, N, k)

    # gather neighbor coordinates
    knn_points = torch.gather(
        x.unsqueeze(2).expand(-1, -1, k, -1),          # (B, N, k, 3)
        1,
        knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)    # (B, N, k, 3)
    )
    return knn_points

def knn_points_simple(x, y=None, k=16):
    """
    x: (B, N, 3)
    y: (B, M, 3) or None
    returns: idx of nearest k points in y for each x
    """
    if y is None:
        y = x
    dist = torch.cdist(x, y, p=2)  # (B, N, M)
    knn_idx = dist.topk(k, dim=-1, largest=False).indices  # (B, N, k)
    return knn_idx


def curvature_loss(x_pred, x_gt, k=16):
    """
    x_pred, x_gt: (B, N, 3) point clouds
    
    1. Get local patch for each point in x_pred, x_gt
    2. Estimate curvature per point (via eigenvalue of local covariance)
    3. L2 loss between corresponding curvatures
    """
    assert x_pred.shape == x_gt.shape, "Shape mismatch"
    B, N, _ = x_pred.shape

    # Get knn for predicted and GT
    idx_pred = knn_points_simple(x_pred, k=k)  # (B, N, k)
    idx_gt = knn_points_simple(x_gt, k=k)

    def get_curvature(pc, idx):
        B, N, k = idx.shape
        patch = torch.gather(
            pc.unsqueeze(2).expand(-1, -1, k, -1),
            1,
            idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # (B, N, k, 3)

        center = pc.unsqueeze(2)  # (B, N, 1, 3)
        diffs = patch - center  # (B, N, k, 3)

        # Covariance matrix per point (B, N, 3, 3)
        cov = diffs.transpose(2, 3) @ diffs / (k - 1)

        # Eigen decomposition for curvature (smallest eigenvalue)
        curvature = torch.linalg.eigvalsh(cov)  # (B, N, 3)
        min_curvature = curvature[:, :, 0]  # (B, N)
        return min_curvature

    curv_pred = get_curvature(x_pred, idx_pred)
    curv_gt = get_curvature(x_gt, idx_gt)

    loss = F.l1_loss(curv_pred, curv_gt)
    return loss

def estimate_normals(points, k=20):
    """
    Estimate normals from point cloud using PCA on KNN.
    Args:
        points: [B, N, 3]
    Returns:
        normals: [B, N, 3] (unit vector)
    """
    B, N, _ = points.shape
    idx = knn_points_simple(points, k=k)  # [B, N, k]

    # Gather neighbors: [B, N, k, 3]
    neighbors = torch.gather(
        points.unsqueeze(2).expand(-1, -1, k, -1),  # [B, N, k, 3]
        1,
        idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    )

    centered = neighbors - points.unsqueeze(2)  # [B, N, k, 3]
    cov = centered.transpose(2, 3) @ centered  # [B, N, 3, 3]
    cov = cov / k

    normals = []
    for b in range(B):
        e, v = torch.linalg.eigh(cov[b])  # [N, 3], [N, 3, 3]
        n = v[:, :, 0]  # eigenvector with smallest eigenvalue
        normals.append(n)

    normals = torch.stack(normals, dim=0)  # [B, N, 3]
    normals = torch.nn.functional.normalize(normals, dim=-1)
    return normals

def normal_loss(pred_points, gt_points, k=20):
    """
    Align predicted normals and GT normals using cosine similarity.
    Args:
        pred_points: [B, N, 3]
        gt_points:   [B, N, 3]
    Returns:
        scalar loss
    """
    pred_normals = estimate_normals(pred_points, k)  # [B, N, 3]
    gt_normals = estimate_normals(gt_points, k)      # [B, N, 3]

    # Cosine similarity loss (1 - cos_sim)
    cos_sim = torch.sum(pred_normals * gt_normals, dim=-1)  # [B, N]
    loss = 1.0 - cos_sim.abs()  # ignore sign
    return loss.mean()

def laplacian_smoothness_loss(x, k=16):
    idx = knn_points_simple(x, k=k)
    knn_pts = torch.gather(
        x.unsqueeze(2).expand(-1, -1, k, -1),
        1,
        idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    )
    laplacian = x - knn_pts.mean(dim=2)
    return (laplacian ** 2).mean()

def soft_chamfer_distance(x, y, tau=0.01):
    """
    Soft Chamfer Distance between two point clouds x and y using softmin weights.
    Args:
        x: (B, N, 3) - predicted point cloud
        y: (B, M, 3) - target point cloud
        tau: temperature for softmin (smaller = closer to hard min)
    Returns:
        loss: scalar Chamfer distance
    """
    B, N, _ = x.shape
    _, M, _ = y.shape

    # Compute pairwise squared distances: (B, N, M)
    x_sq = x.pow(2).sum(dim=2, keepdim=True)  # (B, N, 1)
    y_sq = y.pow(2).sum(dim=2, keepdim=True).transpose(1, 2)  # (B, 1, M)
    inner = torch.bmm(x, y.transpose(1, 2))  # (B, N, M)
    dist = x_sq - 2 * inner + y_sq  # (B, N, M)

    # Softmin over y for each x (direction: x → y)
    weights_x = F.softmax(-dist / tau, dim=2)  # (B, N, M)
    soft_dist_x = (weights_x * dist).sum(dim=2).mean()

    # Softmin over x for each y (direction: y → x)
    weights_y = F.softmax(-dist / tau, dim=1)  # (B, N, M)
    soft_dist_y = (weights_y * dist).sum(dim=1).mean()

    return soft_dist_x + soft_dist_y

class LION(object):
    def __init__(self, cfg):
        # self.vae = VAE(cfg).cuda()
        self.vae = VAE(cfg)
        GlobalPrior = import_model(cfg.latent_pts.style_prior)
        global_prior = GlobalPrior(cfg.sde, cfg.latent_pts.style_dim, cfg)
        local_prior = LocalPrior(cfg.sde, cfg.shapelatent.latent_dim, cfg)
        self.priors = torch.nn.ModuleList([global_prior, local_prior])
        self.scheduler = DDPMScheduler(clip_sample=False,
                                       beta_start=cfg.ddpm.beta_1, beta_end=cfg.ddpm.beta_T, beta_schedule=cfg.ddpm.sched_mode,
                                       num_train_timesteps=cfg.ddpm.num_steps, variance_type=cfg.ddpm.model_var_type)
        self.diffusion = DiffusionDiscretized(None, None, cfg)
        # self.load_model(cfg)

    def to(self, device):
        self.vae.to(device)
        self.priors.to(device)
        print(f"LION's components (VAE, Priors) moved to {device}.")
        return self

    def load_model(self, model_path):
        # model_path = cfg.ckpt.path
        ckpt = torch.load(model_path)
        self.priors.load_state_dict(ckpt['dae_state_dict'])
        self.vae.load_state_dict(ckpt['vae_state_dict'])
        print(f'INFO finish loading from {model_path}')
    
    @torch.no_grad()
    def sample(self, num_samples=10, clip_feat=None, save_img=False):
        self.scheduler.set_timesteps(1000, device='cuda')
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]
        assert(not local_prior.mixed_prediction and not global_prior.mixed_prediction)
        sampled_list = []
        output_dict = {}

        # start sample global prior
        x_T_shape = [num_samples] + latent_shape[0]
        x_noisy = torch.randn(size=x_T_shape, device='cuda')
        condition_input = None
        for i, t in enumerate(tqdm(timesteps, desc="Sampling Global Prior")):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)
            noise_pred = global_prior(x=x_noisy, t=t_tensor.float(), 
                    condition_input=condition_input, clip_feat=clip_feat)
            x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample
        sampled_list.append(x_noisy)
        output_dict['z_global'] = x_noisy

        condition_input = x_noisy
        condition_input = self.vae.global2style(condition_input)

        # start sample local prior
        x_T_shape = [num_samples] + latent_shape[1]
        x_noisy = torch.randn(size=x_T_shape, device='cuda')

        for i, t in enumerate(tqdm(timesteps, desc="Sampling Local Prior")):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)
            noise_pred = local_prior(x=x_noisy, t=t_tensor.float(), 
                    condition_input=condition_input, clip_feat=clip_feat)
            x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample
        sampled_list.append(x_noisy)
        output_dict['z_local'] = x_noisy

        # decode the latent
        output = self.vae.sample(num_samples=num_samples, decomposed_eps=sampled_list)
        if save_img:
            out_name = plot_points(output, "./tmp/tmp.png")
            print(f'INFO save plot image at {out_name}')
        output_dict['points'] = output
        return output_dict

    def dps_sample(self,
                   y: torch.Tensor,
                   loss_fn: torch.nn.Module,
                   guidance_start_t: int,
                   guidance_stop_t: int,
                   guidance_scale: float,
                   guidance_scheduler: str,
                   num_samples: int = 1,
                   forward_model=None,
                   clip_feat=None,
                   debug: bool = False):
        """
        Performs guided sampling using Diffusion Policy Gradient (DPS).
        """
        # 1. Setup
        n_timesteps = 1000
        self.scheduler.set_timesteps(n_timesteps, device=y.device)
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]
        output_dict = {}

        # 2. Global Prior Sampling (Unconditional)
        x_T_shape_global = [num_samples] + latent_shape[0]
        z_global = torch.randn(size=x_T_shape_global, device=y.device)
        with torch.no_grad():
            for t in tqdm(timesteps, desc="Sampling Global Prior"):
                t_tensor = torch.full((num_samples,), t + 1, device=y.device, dtype=torch.long)
                noise_pred_global = global_prior(x=z_global, t=t_tensor.float(), clip_feat=clip_feat)
                z_global = self.scheduler.step(noise_pred_global, t, z_global).prev_sample
        
        output_dict['z_global'] = z_global
        condition_input = self.vae.global2style(z_global)

        # 3. Local Prior Sampling with DPS Guidance
        x_T_shape_local = [num_samples] + latent_shape[1]
        z_local = torch.randn(size=x_T_shape_local, device=y.device)
        
        assert guidance_stop_t <= guidance_start_t
        is_guidance_active = guidance_scale > 0 and guidance_start_t > 0
        
        desc = f"DPS Sampling ({guidance_scheduler} guidance)" if is_guidance_active else "Unconditional Sampling"
        
        grad_list = []
        
        for i, t in enumerate(tqdm(timesteps, desc=desc)):
            t_tensor = torch.full((num_samples,), t + 1, device=y.device, dtype=torch.long)
            
            apply_guidance_this_step = is_guidance_active and t < guidance_start_t and t > guidance_stop_t

            if apply_guidance_this_step:
                # --- Guidance Step ---
                with torch.enable_grad():
                    z_local_grad = z_local.detach().requires_grad_(True)
                    # z_global_grad = z_global.detach().requires_grad_(True)

                    # condition_input_grad = self.vae.global2style(z_global_grad)
                    
                    # noise_pred_local = local_prior(x=z_local_grad, t=t_tensor.float(), condition_input=condition_input_grad, clip_feat=clip_feat)
                    noise_pred_local = local_prior(x=z_local_grad, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    z0_pred_local = (z_local_grad - beta_prod_t.sqrt() * noise_pred_local) / alpha_prod_t.sqrt()
                    
                    # x0_pred = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global_grad, z0_pred_local])
                    x0_pred = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z0_pred_local])
                    
                    y_pred = forward_model.forward(x0_pred)
                    
                    
                    # y_pred_norm = normalize_pointcloud(y_pred)
                    # y_norm = normalize_pointcloud(y)

                    # pred_voxel = pointcloud_to_voxel(y_pred_norm, res=512)
                    # gt_voxel = pointcloud_to_voxel(y_norm, res=512)

                    # loss_vox = F.binary_cross_entropy(pred_voxel, gt_voxel)
                    # breakpoint()
                    # loss_smooth = patchwise_chamfer(y_pred,y,k=8)
                    # loss_smooth = curvature_loss(y_pred, y, k=16)
                    
                    # loss_smooth = laplacian_smoothness_loss(y_pred, k=64)
                    # loss_smooth = normal_loss(y_pred, y, k=16)

                    # breakpoint()
                    # if i<300:
                    #     loss_cf = patchwise_chamfer(y_pred,y,k=4)
                    # elif i >=300 and i <600:
                    #     loss_cf = patchwise_chamfer(y_pred,y,k=16)
                    # # elif i >=600 and i <700:
                    # #     loss_cf = patchwise_chamfer(y_pred,y,k=64)
                    # else:
                    loss_cf = loss_fn(y_pred, y)
                    # loss_emd = torch.mean(earth_mover_distance(y_pred, y, transpose=False))
                    
                    if i<700:
                        loss = loss_cf 
                    else:
                            # y_pred_norm = normalize_pointcloud(y_pred)
                            # y_norm = normalize_pointcloud(y)

                            # pred_voxel = pointcloud_to_voxel(y_pred_norm, res=256)
                            # gt_voxel = pointcloud_to_voxel(y_norm, res=256)

                            # loss_smooth = F.binary_cross_entropy(pred_voxel, gt_voxel)
                            # smooth_weight = 0.003
                            # loss_rep = apply_repulsion_loss(y_pred,k=4)
                            # loss_smooth = apply_laplacian_smoothness(y_pred,k=16)
                        # loss_smooth = apply_patchwise_chamfer(y_pred, y, k = 4)
                        # loss_smooth = torch.mean(earth_mover_distance(y_pred, y, transpose=False))
                        loss_smooth = voxel_bce_loss(y_pred,y,res=16)
                        
                        smooth_weight = 0.0001
                        # rep_weight = 0.001
                        # breakpoint()
                        loss = loss_cf + loss_smooth * smooth_weight 
                        # loss = loss_smooth * smooth_weight 


                    # loss = loss_cf 

                    grad = torch.autograd.grad(loss, z_local_grad)[0]

                    
                    # grad = indexwise_smooth_gradient(grad = grad.squeeze(-1))
                    # grad = grad.unsqueeze(-1)
        
                    
                    

                with torch.no_grad():
                    noise_pred_detached = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    prev_sample = self.scheduler.step(noise_pred_detached, t, z_local).prev_sample
                                        
                    norm_gradient_local = grad / torch.sqrt(loss.detach())
                    # loss_norm = torch.norm(y - y_pred, dim=-1, keepdim=True)
                    # scale = 1 / (loss_norm + 1e-6)
                    # breakpoint()
                    # norm_gradient_local = grad *scale
                    
                    if guidance_scheduler == 'linear':
                        scale_t = (t - guidance_stop_t) / (guidance_start_t - guidance_stop_t)
                        guidance_scale_t = guidance_scale * scale_t
                    if guidance_scheduler == 'linear+const':
                        if t <600 :
                            guidance_scale_t = 15
                        else:
                            guidance_scale_t = guidance_scale
                        
                            # scale_t = (t - guidance_stop_t) / (guidance_start_t - guidance_stop_t)
                            # guidance_scale_t = guidance_scale * scale_t 

                    else:  # 'constant'
                        guidance_scale_t = guidance_scale
                    
                    z_local = prev_sample - guidance_scale_t * norm_gradient_local
                    
                    
                    # z_local = percentile_clip(z_local, lower=0.5, upper=99.5)

                    # z_local = prev_sample
                    # z_global = z_global - guidance_scale_t * norm_gradient_global * 0.01

            else:
                # --- Unconditional Step ---
                with torch.no_grad():
                    noise_pred_local = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    z_local = self.scheduler.step(noise_pred_local, t, z_local).prev_sample

            # --- Debugging and Visualization ---
            if debug and (i % 100 == 0 or i == len(timesteps) - 1 ):
                with torch.no_grad():
                    grad_list.append(grad)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    z0_pred_local = (z_local - beta_prod_t.sqrt() * noise_pred_local) / alpha_prod_t.sqrt()
                    
                    # x0_pred = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global_grad, z0_pred_local])
                    x0_pred = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z0_pred_local])

                    # if i > 900:
                    #     breakpoint()
                    z_local_norm = torch.linalg.norm(z_local).item()
                    z_local_std = torch.std(z_local).item()
                    x0_pred_std = torch.std(x0_pred).item()
                    x0_pred_abs = torch.abs(x0_pred)
                    x0_pred_abs_max = torch.max(x0_pred_abs).item()

                    logging.info(
                        f"\nDebug [Step {i:04d}/{len(timesteps)}]"
                        f" - CD Error: {loss:.4f}"
                        f" | z_local Norm: {z_local_norm:.4f}"
                        f" | z_local Std: {z_local_std:.4f}"
                        f" | x0_pred Std: {x0_pred_std:.4f}"
                        f" | x0_pred Max: {x0_pred_abs_max:.4f}"
                    )
                    
                    # Save intermediate reconstruction
                    plot_points(x0_pred, f"./vis_chang/recon_step_{i:04d}.png")
                    visualize_single_timestep_gradient(z_local, timestep = i, save_path=f"./vis_chang/grad_heatmap_{i:04d}.png")

                
        output_dict['z_local'] = z_local

        # 4. Final Decoding
        with torch.no_grad():
            output = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z_local])
        output_dict['points'] = output
        return output_dict
        
    def plsd_sample(self,
                    y: torch.Tensor,
                    loss_fn: torch.nn.Module,
                    guidance_start_t: int,
                    guidance_stop_t: int,
                    guidance_scale: float,
                    consistency_scale: float,
                    guidance_scheduler: str,
                    num_samples: int = 1,
                    forward_model=None,
                    clip_feat=None,
                    debug: bool = False):
        """
        Performs guided sampling using Proximal Score-based Likelihood Denoising (PSLD).
        This involves a measurement guidance step and a consistency guidance step.
        """
        # 1. Setup
        n_timesteps = 1000
        self.scheduler.set_timesteps(n_timesteps, device=y.device)
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]
        output_dict = {}

        if debug:
            debug_cd_loss = chamfer_3DDist()

        # 2. Global Prior Sampling (Unconditional)
        # This samples the global style vector `z_global` which conditions the local prior.
        x_T_shape_global = [num_samples] + latent_shape[0]
        z_global = torch.randn(size=x_T_shape_global, device=y.device)
        with torch.no_grad():
            for t in tqdm(timesteps, desc="Sampling Global Prior"):
                t_tensor = torch.full((num_samples,), t + 1, device=y.device, dtype=torch.long)
                noise_pred_global = global_prior(x=z_global, t=t_tensor.float(), clip_feat=clip_feat)
                z_global = self.scheduler.step(noise_pred_global, t, z_global).prev_sample
        
        output_dict['z_global'] = z_global
        condition_input = self.vae.global2style(z_global)

        # 3. Local Prior Sampling with PSLD Guidance
        x_T_shape_local = [num_samples] + latent_shape[1]
        z_local = torch.randn(size=x_T_shape_local, device=y.device)
        
        assert guidance_stop_t <= guidance_start_t
        is_guidance_active = guidance_start_t > 0 or guidance_stop_t < 1000
        
        desc = f"PSLD Sampling ({guidance_scheduler} guidance)" if is_guidance_active else "Unconditional Sampling"

        for i, t in enumerate(tqdm(timesteps, desc=desc)):
            t_tensor = torch.full((num_samples,), t + 1, device=y.device, dtype=torch.long)
            
            # Determine if guidance should be applied at the current timestep
            apply_guidance_this_step = is_guidance_active and t < guidance_start_t and t > guidance_stop_t

            if apply_guidance_this_step:
                # --- Guidance Step ---
                with torch.enable_grad():
                    z_local_grad = z_local.detach().requires_grad_(True)
                    
                    # Predict noise to estimate x0
                    noise_pred_local = local_prior(x=z_local_grad, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    z0_pred = (z_local_grad - beta_prod_t.sqrt() * noise_pred_local) / alpha_prod_t.sqrt()
                    
                    # Decode estimated x0 and calculate loss
                    x0_pred = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z0_pred])
                    
                    # 1. Measurement Loss: Guides the output towards the measurement `y`.
                    y_pred = forward_model.forward(x0_pred)
                    
                    # loss_measurement = loss_fn(y_pred, y)

                    dist1, dist2, _, _ = chamfer_3DFunction.apply(y_pred, y)
                    loss_measurement = torch.mean(dist1) + torch.mean(dist2)

                    # 2. Consistency Loss: Enforces data consistency in the latent space.
                    y_backprojected = forward_model.transpose(y)
                    x0_pred_backprojected = forward_model.transpose(y_pred)
                    consistency_target_x = y_backprojected + (x0_pred - x0_pred_backprojected)
                    
                    consistency_target_z = self.vae.get_local_posterior_mean(consistency_target_x.detach(), condition_input)
                    
                    loss_consistency = torch.nn.functional.mse_loss(z0_pred, consistency_target_z.detach())

                    grad_measurement = torch.autograd.grad(loss_measurement, z_local_grad, retain_graph=True)[0]
                    grad_consistency = torch.autograd.grad(loss_consistency, z_local_grad)[0]

                with torch.no_grad():
                    # Perform the standard reverse diffusion step (prediction)
                    prev_sample = self.scheduler.step(noise_pred_local.detach(), t, z_local).prev_sample
                    
                    # Normalize gradients
                    normalized_grad_measurement = grad_measurement / (torch.sqrt(loss_measurement.detach()) + 1e-8)
                    normalized_grad_consistency = grad_consistency / (torch.sqrt(loss_consistency.detach()) + 1e-8)
                    
                    # Gradient Clipping
                    # normalized_grad_measurement = torch.clip(normalized_grad_measurement, min=-0.1, max=0.1)
                    # normalized_grad_consistency = torch.clip(normalized_grad_consistency, min=-0.1, max=0.1)
                    
                    # Calculate time-dependent guidance scales
                    if guidance_scheduler == 'linear':
                        scale_t = (t - guidance_stop_t) / (guidance_start_t - guidance_stop_t)
                        guidance_scale_t = guidance_scale * scale_t
                        consistency_scale_t = consistency_scale * scale_t
                    else:  # 'constant'
                        guidance_scale_t = guidance_scale
                        consistency_scale_t = consistency_scale
                    
                    # Apply guidance corrections for both terms
                    z_local = prev_sample \
                            - guidance_scale_t * normalized_grad_measurement \
                            - consistency_scale_t * normalized_grad_consistency
            
            else:
                # --- Unconditional Step ---
                with torch.no_grad():
                    noise_pred_local = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    z_local = self.scheduler.step(noise_pred_local, t, z_local).prev_sample
            
            # --- Debugging and Visualization ---
            if debug and (i % 100 == 0 or i == len(timesteps) - 1):
                with torch.no_grad():
                    z_local_norm = torch.linalg.norm(z_local).item()
                    z_local_std = torch.std(z_local).item()

                    logging.info(
                        f"\nDebug [Step {i:04d}/{len(timesteps)}]"
                        f" - CD Error: {loss_measurement:.4f}"
                        f" | z_local Norm: {z_local_norm:.4f}"
                        f" | z_local Var: {z_local_std:.4f}"
                    )
                    
                    # Save intermediate reconstruction
                    plot_points(x0_pred, f"./tmp/recon_step_{i:04d}.png")
        
        output_dict['z_local'] = z_local

        # 4. Final Decoding
        with torch.no_grad():
            output = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z_local])
        output_dict['points'] = output
        return output_dict

    def get_mixing_component(self, noise_pred, t):
        # usage:
        # if global_prior.mixed_prediction:
        #     mixing_component = self.get_mixing_component(noise_pred, t)
        #     coeff = torch.sigmoid(global_prior.mixing_logit)
        #     noise_pred = (1 - coeff) * mixing_component + coeff * noise_pred

        alpha_bar = self.scheduler.alphas_cumprod[t]
        one_minus_alpha_bars_sqrt = np.sqrt(1.0 - alpha_bar)
        return noise_pred * one_minus_alpha_bars_sqrt
