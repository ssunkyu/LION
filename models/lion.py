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
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

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
                   guidance_scale: float,
                   guidance_scheduler: str,
                   guidance_start_t: int,
                   num_samples=1,
                   forward_model=None,
                   clip_feat=None,
                   save_img=False,
                   debug=False):
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

        # For debug logging
        if debug:
            debug_cd_loss = chamfer_3DDist()

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

        # 3. Local Prior Sampling with optional DPS Guidance
        x_T_shape_local = [num_samples] + latent_shape[1]
        z_local = torch.randn(size=x_T_shape_local, device=y.device)
        
        is_guidance_active = guidance_scale > 0 and guidance_start_t > 0
        desc = f"DPS Sampling ({guidance_scheduler} guidance)" if is_guidance_active else "Unconditional Sampling"

        for i, t in enumerate(tqdm(timesteps, desc=desc)):
            t_tensor = torch.full((num_samples,), t + 1, device=y.device, dtype=torch.long)
            
            # Determine if guidance should be applied at the current timestep
            apply_guidance_this_step = is_guidance_active and t < guidance_start_t

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
                    y_pred = forward_model.forward(x0_pred) if forward_model is not None else x0_pred
                    loss = loss_fn(y_pred, y)
                    
                    # Get gradient of the loss w.r.t. the latent
                    grad = torch.autograd.grad(loss, z_local_grad)[0]

                with torch.no_grad():
                    # Get the unconditional sample from the scheduler
                    prev_sample = self.scheduler.step(noise_pred_local.detach(), t, z_local).prev_sample
                    
                    # Normalize gradient to use only its direction
                    grad_normalized = grad / (torch.sqrt(loss.detach()) + 1e-8)
                    
                    # Calculate time-dependent guidance scale
                    if guidance_scheduler == 'linear':
                        guidance_scale_t = guidance_scale * (t / guidance_start_t)
                    else: # 'constant'
                        guidance_scale_t = guidance_scale
                    
                    # Apply the centered guidance
                    z_local = prev_sample - guidance_scale_t * grad_normalized
            
            else:
                # --- Unconditional Step ---
                with torch.no_grad():
                    noise_pred_local = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    z_local = self.scheduler.step(noise_pred_local, t, z_local).prev_sample
            
            # --- Debugging and Visualization ---
            if debug and (i % 100 == 0 or i == len(timesteps) - 1):
                with torch.no_grad():
                    # Always predict x0 from the current z_local for logging
                    current_noise_pred = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    z0_pred_debug = (z_local - beta_prod_t.sqrt() * current_noise_pred) / alpha_prod_t.sqrt()
                    x0_pred_debug = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z0_pred_debug])
                    
                    # [요청 3] Log reconstruction error (Chamfer Distance) instead of grad norm
                    y_pred_debug = forward_model.forward(x0_pred_debug)
                    dist1, dist2, _, _ = debug_cd_loss(y_pred_debug, y)
                    cd_error = torch.mean(dist1) + torch.mean(dist2)

                    logging.info(f"\nDebug [Step {i:04d}/{len(timesteps)}] - Reconstruction Error (CD): {cd_error.item():.4f}")
                    
                    # Save intermediate reconstruction
                    plot_points(x0_pred_debug, f"./tmp/recon_step_{i:04d}.png")
        
        output_dict['z_local'] = z_local

        # 4. Final Decoding
        with torch.no_grad():
            output = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z_local])
        output_dict['points'] = output
        return output_dict
    
    def plsd_sample(self,
                    y: torch.Tensor,
                    loss_fn: torch.nn.Module,
                    guidance_scale: float,
                    guidance_scheduler: str,
                    guidance_start_t: int,
                    num_samples=1,
                    forward_model=None,
                    clip_feat=None,
                    save_img=False,
                    debug=False):
        """
        Performs guided sampling using Diffusion Policy Gradient (PLSD).
        """
        # 1. Setup
        n_timesteps = 1000
        self.scheduler.set_timesteps(n_timesteps, device=y.device)
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]
        output_dict = {}

        # For debug logging
        if debug:
            debug_cd_loss = chamfer_3DDist()

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

        # 3. Local Prior Sampling with optional DPS Guidance
        x_T_shape_local = [num_samples] + latent_shape[1]
        z_local = torch.randn(size=x_T_shape_local, device=y.device)
        
        is_guidance_active = guidance_scale > 0 and guidance_start_t > 0
        desc = f"DPS Sampling ({guidance_scheduler} guidance)" if is_guidance_active else "Unconditional Sampling"

        for i, t in enumerate(tqdm(timesteps, desc=desc)):
            t_tensor = torch.full((num_samples,), t + 1, device=y.device, dtype=torch.long)
            
            # Determine if guidance should be applied at the current timestep
            apply_guidance_this_step = is_guidance_active and t < guidance_start_t

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
                    y_pred = forward_model.forward(x0_pred) if forward_model is not None else x0_pred
                    loss = loss_fn(y_pred, y)
                    
                    # Get gradient of the loss w.r.t. the latent
                    grad = torch.autograd.grad(loss, z_local_grad)[0]

                with torch.no_grad():
                    # Get the unconditional sample from the scheduler
                    prev_sample = self.scheduler.step(noise_pred_local.detach(), t, z_local).prev_sample
                    
                    # Normalize gradient to use only its direction
                    grad_normalized = grad / (torch.sqrt(loss.detach()) + 1e-8)
                    
                    # Calculate time-dependent guidance scale
                    if guidance_scheduler == 'linear':
                        guidance_scale_t = guidance_scale * (t / guidance_start_t)
                    else: # 'constant'
                        guidance_scale_t = guidance_scale
                    
                    # Apply the centered guidance
                    z_local = prev_sample - guidance_scale_t * grad_normalized
            
            else:
                # --- Unconditional Step ---
                with torch.no_grad():
                    noise_pred_local = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    z_local = self.scheduler.step(noise_pred_local, t, z_local).prev_sample
            
            # --- Debugging and Visualization ---
            if debug and (i % 100 == 0 or i == len(timesteps) - 1):
                with torch.no_grad():
                    # Always predict x0 from the current z_local for logging
                    current_noise_pred = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    z0_pred_debug = (z_local - beta_prod_t.sqrt() * current_noise_pred) / alpha_prod_t.sqrt()
                    x0_pred_debug = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z0_pred_debug])
                    
                    # [요청 3] Log reconstruction error (Chamfer Distance) instead of grad norm
                    y_pred_debug = forward_model.forward(x0_pred_debug)
                    dist1, dist2, _, _ = debug_cd_loss(y_pred_debug, y)
                    cd_error = torch.mean(dist1) + torch.mean(dist2)

                    logging.info(f"\nDebug [Step {i:04d}/{len(timesteps)}] - Reconstruction Error (CD): {cd_error.item():.4f}")
                    
                    # Save intermediate reconstruction
                    plot_points(x0_pred_debug, f"./tmp/recon_step_{i:04d}.png")
        
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
