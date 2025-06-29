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

class LION(object):
    def __init__(self, cfg):
        # self.vae = VAE(cfg).cuda()
        self.vae = VAE(cfg)
        GlobalPrior = import_model(cfg.latent_pts.style_prior)
        # global_prior = GlobalPrior(cfg.sde, cfg.latent_pts.style_dim, cfg).cuda()
        global_prior = GlobalPrior(cfg.sde, cfg.latent_pts.style_dim, cfg)
        # local_prior = LocalPrior(cfg.sde, cfg.shapelatent.latent_dim, cfg).cuda()
        local_prior = LocalPrior(cfg.sde, cfg.shapelatent.latent_dim, cfg)
        self.priors = torch.nn.ModuleList([global_prior, local_prior])
        self.scheduler = DDPMScheduler(clip_sample=False,
                                       beta_start=cfg.ddpm.beta_1, beta_end=cfg.ddpm.beta_T, beta_schedule=cfg.ddpm.sched_mode,
                                       num_train_timesteps=cfg.ddpm.num_steps, variance_type=cfg.ddpm.model_var_type)
        self.diffusion = DiffusionDiscretized(None, None, cfg)
        # self.load_model(cfg)

    def to(self, device):
        """
        Moves all internal PyTorch models (VAE and Priors) to the specified device.

        Args:
            device (str or torch.device): The device to move the models to ('cuda', 'cpu', etc.).
        
        Returns:
            self: The instance of the LION class.
        """
        self.vae.to(device)
        # self.priors is an nn.ModuleList, so calling .to() on it moves all contained modules.
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
    
    def dps_sample(self, y: torch.Tensor, loss_fn, guidance_scale: float, num_samples=1,
                   forward_model=None, clip_feat=None, save_img=False):
        """
        DPS + Chamfer distance
        
        Args:
            y (torch.Tensor): measurements.
            guidance_scale (float): reconstruction guidance scale.
            forward_model : measurement operator.
        """
        self.scheduler.set_timesteps(1000, device='cuda')
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]
        assert(not local_prior.mixed_prediction and not global_prior.mixed_prediction)
        
        output_dict = {}

        # start sample global prior
        x_T_shape_global = [num_samples] + latent_shape[0]
        z_global = torch.randn(size=x_T_shape_global, device='cuda')
        for i, t in enumerate(tqdm(timesteps, desc="Sampling Global Prior")):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t + 1)
            noise_pred_global = global_prior(x=z_global, t=t_tensor.float(), clip_feat=clip_feat)
            z_global = self.scheduler.step(noise_pred_global, t, z_global).prev_sample
        output_dict['z_global'] = z_global
        
        condition_input = z_global
        condition_input = self.vae.global2style(condition_input)

        # start sample local prior
        x_T_shape_local = [num_samples] + latent_shape[1]
        z_local = torch.randn(size=x_T_shape_local, device='cuda')
        for i, t in enumerate(tqdm(timesteps, desc="Sampling Local Prior")):
            with torch.enable_grad():
                z_local_grad = z_local.detach().requires_grad_(True)
                
                t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t + 1)
                noise_pred_local = local_prior(x=z_local_grad, t=t_tensor.float(), 
                                               condition_input=condition_input, clip_feat=clip_feat)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                z0_pred = (z_local_grad - beta_prod_t.sqrt() * noise_pred_local) / alpha_prod_t.sqrt()
                
                # --- THIS IS THE FIX ---
                # Pass latents as a list to the 'decomposed_eps' argument.
                x0_pred = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z0_pred])
                # ----------------------
                
                if forward_model is not None:
                    y_pred = forward_model.forward(x0_pred)
                else:
                    y_pred = x0_pred
                
                dist1, dist2, _, _ = loss_fn(y_pred, y)
                loss = torch.mean(dist1) + torch.mean(dist2)
                
                grad = torch.autograd.grad(loss, z_local_grad)[0]

            # score gradient step
            z_local_prev = self.scheduler.step(noise_pred_local, t, z_local).prev_sample
            
            # reconstruction gradient step
            z_local = z_local_prev - guidance_scale * grad
        
        output_dict['z_local'] = z_local

        # Final decoding of the latent variables
        output = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z_local])
        
        if save_img:
            out_name = plot_points(output, "./tmp/tmp.png")
            print(f'INFO save plot image at {out_name}')
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
