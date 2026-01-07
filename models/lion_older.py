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
from third_party.Density_aware_Chamfer_Distance.utils_v2.model_utils import calc_dcd, calc_scd
from utils.demo_utils import plot_guidance_schedule, log_and_visualize_debug_info
import torch.nn.functional as F

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
                   guidance_stop_t: int,
                   num_samples=1,
                   forward_model=None,
                   clip_feat=None,
                   save_img=False,
                   debug=False):
        """
        Performs guided sampling using Diffusion Policy Gradient (DPS).
        """
        # --- 1. 초기 설정 ---
        n_timesteps = 1000
        self.scheduler.set_timesteps(n_timesteps, device=y.device)
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]

        # History-tracking lists
        output_dict = {}
        loss_history_ = []
        scale_history_ = []
        loss_history = []
        scale_history = []

        # Ground Truth(GT) 잠재 변수 추출
        z_global_gt = self.vae.encode_global(y).sample()[0]
        style = self.vae.global2style(z_global_gt)
        z_local_gt = self.vae.encode_local(y, style).sample()[0]

        assert guidance_stop_t <= guidance_start_t
        is_guidance_active = guidance_scale > 0 and guidance_start_t > 0
        desc_global = f"DPS Global Sampling ({guidance_scheduler})" if is_guidance_active else "Unconditional Global Sampling"
        
        # --- 2. Global Prior 샘플링 ---
        x_T_shape_global = [num_samples] + latent_shape[0]
        z_global = torch.randn(size=x_T_shape_global, device=y.device)

        with torch.no_grad():
            for t in tqdm(timesteps, desc="Sampling Global Prior"):
                t_tensor = torch.full((num_samples,), t + 1, device=y.device, dtype=torch.long)
                noise_pred_global = global_prior(x=z_global, t=t_tensor.float(), clip_feat=clip_feat)
                z_global = self.scheduler.step(noise_pred_global, t, z_global).prev_sample
                
        # z_global = self.vae.encode_global(y, 'chair').sample()[0].unsqueeze(-1).unsqueeze(-1)
        # z_global_gt = self.vae.encode_global(y,'chair').sample()[0].unsqueeze(-1).unsqueeze(-1)
        # z_global = self.vae.encode_global(y, 'chair').mean()[0].unsqueeze(0)
        # # z_global_gt = self.vae.encode_global(y).mean()[0].unsqueeze(0)

        # for i, t in enumerate(tqdm(timesteps, desc=desc_global)):
        #     t_tensor = torch.full((num_samples,), t + 1, device=y.device, dtype=torch.long)
        #     apply_guidance_this_step = is_guidance_active and t < guidance_start_t and t > guidance_stop_t
            
        #     # 루프 시작 시 변수 초기화
        #     current_loss = 0.0
        #     guidance_scale_t = 0.0
        #     x0_pred = None

        #     if apply_guidance_this_step:
        #         with torch.enable_grad():
        #             z_global_grad = z_global.detach().requires_grad_(True)
                    
        #             # 예측 및 Loss 계산
        #             noise_pred_global_grad = global_prior(x=z_global_grad, t=t_tensor.float(), clip_feat=clip_feat)
        #             alpha_prod_t = self.scheduler.alphas_cumprod[t]
        #             beta_prod_t = 1 - alpha_prod_t
        #             z0_pred_global = (z_global_grad - beta_prod_t.sqrt() * noise_pred_global_grad) / alpha_prod_t.sqrt()
        #             x0_pred = self.vae.sample(num_samples=num_samples, decomposed_eps=[z0_pred_global, z_local_gt])
                    
        #             y_pred = forward_model.forward(x0_pred)
        #             loss = loss_fn(y_pred, y)
        #             grad = torch.autograd.grad(loss, z_global_grad)[0]
                
        #         current_loss = loss.item()
        #         # guidance_scale_t = 0.3
        #         guidance_scale_t = 10 - 0.01 * t.item()
        #         # guidance_scale_t = 0.01 * t.item()

        #         with torch.no_grad():
        #             # 노이즈 재예측 및 샘플 업데이트
        #             noise_pred_global = global_prior(x=z_global, t=t_tensor.float(), clip_feat=clip_feat)
        #             prev_sample = self.scheduler.step(noise_pred_global, t, z_global).prev_sample
                    
        #             # 그래디언트를 이용한 가이던스 적용
        #             norm_gradient_global = grad / torch.sqrt(loss.detach())
        #             z_global = prev_sample - guidance_scale_t * norm_gradient_global
        #     else:
        #         with torch.no_grad():
        #             noise_pred_global = global_prior(x=z_global, t=t_tensor.float(), clip_feat=clip_feat)
        #             z_global = self.scheduler.step(noise_pred_global, t, z_global).prev_sample
            
        #     loss_history_.append(current_loss)
        #     scale_history_.append(guidance_scale_t)
            
        #     if debug and (i % 100 == 0 or i == len(timesteps) - 1) and x0_pred is not None:
        #         with torch.no_grad():
        #             log_and_visualize_debug_info(self, i, len(timesteps), t.item(), z_local_gt, z_global, noise_pred_global, current_loss, x0_pred)

        output_dict['z_global'] = z_global
        condition_input = self.vae.global2style(z_global)
        # condition_input_zero = torch.zeros_like(condition_input)

        # --- 3. Local Prior 샘플링 ---
        x_T_shape_local = [num_samples] + latent_shape[1]
        z_local = torch.randn(size=x_T_shape_local, device=y.device)
        desc_local = f"DPS Local Sampling ({guidance_scheduler})" if is_guidance_active else "Unconditional Local Sampling"

        for i, t in enumerate(tqdm(timesteps, desc=desc_local)):
            t_tensor = torch.full((num_samples,), t + 1, device=y.device, dtype=torch.long)
            apply_guidance_this_step = is_guidance_active and t < guidance_start_t and t > guidance_stop_t

            # 루프 시작 시 변수 초기화
            current_loss = 0.0
            guidance_scale_t = 0.0
            x0_pred = None

            if apply_guidance_this_step:
                with torch.enable_grad():
                    z_local_grad = z_local.detach().requires_grad_(True)
                    # condition_input_grad = condition_input.detach().requires_grad_(True)
                    
                    # 예측 및 Loss 계산
                    noise_pred_local_grad = local_prior(x=z_local_grad, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    # noise_pred_local_grad = local_prior(x=z_local_grad, t=t_tensor.float(), condition_input=condition_input_zero, clip_feat=clip_feat)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    z0_pred_local = (z_local_grad - beta_prod_t.sqrt() * noise_pred_local_grad) / alpha_prod_t.sqrt()
                    x0_pred = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z0_pred_local])
                    
                    y_pred = forward_model.forward(x0_pred)
                    # loss, cd_p, cd_t, f1, dist1, dist2, idx1, idx2 = calc_scd(y_pred, y, alpha=0.01, n_lambda=0.2, return_raw=True)
                    loss, cd_p, cd_t = calc_scd(y_pred, y, alpha=0.01, n_lambda=0.2, return_raw=False)
                    # loss = loss_fn(y_pred.cuda(), y.cuda())
                    # loss_3 = loss * 0.001 + loss_2
                    # loss = loss + (1 - f1)
                    grad = torch.autograd.grad(loss, z_local_grad)[0]
                    # grad, grad_cond = torch.autograd.grad(
                    #     outputs=loss, # .mean()으로 스칼라화
                    #     inputs=[z_local_grad, condition_input_grad],
                    #     allow_unused=True,
                    # )
                
                current_loss = loss.item()
                # if t < 100: # 마지막 100 스텝에서는 가이던스 적용 안함
                #     guidance_scale_t = 0
                # else:
                #     guidance_scale_t = 10 - 0.01 * t.item()
                # guidance_scale_t = 20 - 0.02 * t.item()
                guidance_scale_t = 0.4 * (1 - 0.001 * t.item())
                # guidance_scale_t = 0.01 * t.item()
                # guidance_scale_t = 2.0
                # if t < 50:
                #     guidance_scale_t = 0

                with torch.no_grad():
                    # 노이즈 재예측 및 샘플 업데이트
                    noise_pred_local = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    # noise_pred_local = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input_zero, clip_feat=clip_feat)
                    prev_sample = self.scheduler.step(noise_pred_local, t, z_local).prev_sample
                    
                    # 그래디언트를 이용한 가이던스 적용
                    norm_gradient_local = grad / torch.sqrt(loss.detach())
                    # grad_normal_direction = (z_local - prev_sample) / (z_local - prev_sample).norm()
                    # dot_product = torch.sum(grad_normal_direction * norm_gradient_local)
                    # norm_gradient_local = norm_gradient_local - dot_product * grad_normal_direction
                    z_local = prev_sample - guidance_scale_t * norm_gradient_local
                    # condition_input = condition_input - guidance_scale_t * grad_cond
                    # z_local = z_local.normal_()
            else: # 가이던스 없는 일반 샘플링
                with torch.no_grad():
                    noise_pred_local = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    # noise_pred_local = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input_zero, clip_feat=clip_feat)
                    z_local = self.scheduler.step(noise_pred_local, t, z_local).prev_sample

                    # 가이던스 없는 스텝의 loss 계산 (디버깅/로깅용)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    z0_pred_local = (z_local - beta_prod_t.sqrt() * noise_pred_local) / alpha_prod_t.sqrt()
                    x0_pred = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z0_pred_local])
                    y_pred = forward_model.forward(x0_pred)
                    loss = loss_fn(y_pred, y)
                    current_loss = loss.item()
                    guidance_scale_t = 0.0
                    
            # --- 주석 처리된 다양한 가이던스 스케줄러 로직 ---
            # if guidance_scheduler == 'linear':
            #     scale_t = (t - guidance_stop_t) / (guidance_start_t - guidance_stop_t)
            #     guidance_scale_t = guidance_scale * scale_t + 100
            # else:  # 'constant'
            #     guidance_scale_t = guidance_scale
            # if t > 950:
            #     guidance_scale_t = 0
            # guidance_schedule = [
            #     (950, 0), (700, 1), (400, 5), (100, 7.5),
            # ]
            # default_guidance_value = 10
            # guidance_scale_t = default_guidance_value
            # for threshold, value in guidance_schedule:
            #     if t > threshold:
            #         guidance_scale_t = value
            #         break

            # History 및 디버그 정보 기록
            loss_history.append(current_loss)
            scale_history.append(guidance_scale_t)
            
            if debug and (i % 100 == 0 or i == len(timesteps) - 1) and x0_pred is not None:
                with torch.no_grad():
                    log_and_visualize_debug_info(self, i, len(timesteps), t.item(), z_local, z_global, noise_pred_local, current_loss, x0_pred)

        # --- 4. 최종 결과 처리 및 반환 ---
        if debug and is_guidance_active:
            fig = plot_guidance_schedule(self, timesteps.cpu().numpy(), loss_history, scale_history)
            output_dict['guidance_schedule_plot'] = fig
            plt.close(fig)

        output_dict['z_local'] = z_local
        
        with torch.no_grad():
            final_output = x0_pred if x0_pred is not None else self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z_local])
            output_dict['points'] = final_output.detach()

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

    def dps_sample2(self,
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
        Performs guided sampling using Diffusion Policy Gradient (DPS)
        with an integrated loop for global and local priors.
        """
        # --- 1. Setup ---
        n_timesteps = 1000
        self.scheduler.set_timesteps(n_timesteps, device=y.device)
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]
        output_dict = {}
        loss_history = []
        scale_history = []

        # --- 2. Initial Latent Noise ---
        x_T_shape_global = [num_samples] + latent_shape[0]
        z_global = torch.randn(size=x_T_shape_global, device=y.device)
        
        x_T_shape_local = [num_samples] + latent_shape[1]
        z_local = torch.randn(size=x_T_shape_local, device=y.device)

        # Ground truth global latent for guidance
        z_global_gt = self.vae.encode_global(y).mean()[0].unsqueeze(0)

        # --- 3. Integrated Sampling Loop ---
        assert guidance_stop_t <= guidance_start_t
        is_guidance_active = guidance_scale > 0 and guidance_start_t > 0
        desc = f"DPS Sampling ({guidance_scheduler} guidance)" if is_guidance_active else "Unconditional Sampling"

        for i, t in enumerate(tqdm(timesteps, desc=desc)):
            t_tensor = torch.full((num_samples,), t, device=y.device, dtype=torch.long)
            apply_guidance_this_step = is_guidance_active and t < guidance_start_t and t > guidance_stop_t
            
            # Define guidance scale for the current timestep
            # NOTE: You can use different schedulers or scales for global and local if needed.
            # Here, we use the same for simplicity.
            # guidance_scale_t = 10 - 0.01 * t.item() if apply_guidance_this_step else 0.0
            guidance_scale_t = 0.5
            
            # --- 3.1 Global Prior Guidance and Sampling ---
            if apply_guidance_this_step:
                with torch.enable_grad():
                    z_global_grad = z_global.detach().requires_grad_(True)
                    
                    noise_pred_global_grad = global_prior(x=z_global_grad, t=t_tensor.float(), clip_feat=clip_feat)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    z0_pred_global = (z_global_grad - beta_prod_t.sqrt() * noise_pred_global_grad) / alpha_prod_t.sqrt()
                    
                    # Using a simple MSE loss for latent guidance
                    global_loss_fn = torch.nn.MSELoss()
                    global_loss = global_loss_fn(z0_pred_global, z_global_gt)
                    grad_global = torch.autograd.grad(global_loss, z_global_grad)[0]
                
                with torch.no_grad():
                    # Get the unconditional noise prediction for the standard DDPM step
                    noise_pred_global = global_prior(x=z_global, t=t_tensor.float(), clip_feat=clip_feat)
                    prev_sample_global = self.scheduler.step(noise_pred_global, t, z_global).prev_sample
                    
                    # Normalize gradient and apply guidance
                    norm_gradient_global = grad_global / torch.sqrt(global_loss.detach() + 1e-8) # Add epsilon for stability
                    z_global = prev_sample_global - guidance_scale_t * norm_gradient_global
            else:
                # Standard unconditional sampling step for global prior
                with torch.no_grad():
                    noise_pred_global = global_prior(x=z_global, t=t_tensor.float(), clip_feat=clip_feat)
                    z_global = self.scheduler.step(noise_pred_global, t, z_global).prev_sample

            # --- 3.2 Local Prior Guidance and Sampling (using the updated z_global) ---
            with torch.no_grad():
                condition_input = self.vae.global2style(z_global)

            if apply_guidance_this_step:
                with torch.enable_grad():
                    z_local_grad = z_local.detach().requires_grad_(True)
                    
                    noise_pred_local_grad = local_prior(x=z_local_grad, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    z0_pred_local = (z_local_grad - beta_prod_t.sqrt() * noise_pred_local_grad) / alpha_prod_t.sqrt()
                    
                    # Decode to get the final output prediction for loss calculation
                    x0_pred = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global.detach(), z0_pred_local])
                    
                    y_pred = forward_model.forward(x0_pred)
                    local_loss = loss_fn(y_pred, y)
                    grad_local = torch.autograd.grad(local_loss, z_local_grad)[0]
                
                with torch.no_grad():
                    # Get the unconditional noise prediction for the standard DDPM step
                    noise_pred_local = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    prev_sample_local = self.scheduler.step(noise_pred_local, t, z_local).prev_sample
                    
                    # Normalize gradient and apply guidance
                    norm_gradient_local = grad_local / torch.sqrt(local_loss.detach() + 1e-8)
                    z_local = prev_sample_local - guidance_scale_t * norm_gradient_local
            else:
                # Standard unconditional sampling step for local prior
                with torch.no_grad():
                    noise_pred_local = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                    z_local = self.scheduler.step(noise_pred_local, t, z_local).prev_sample
            
            # --- 3.3 Logging and Debugging ---
            current_loss = local_loss.item() if apply_guidance_this_step else 0.0
            loss_history.append(current_loss)
            scale_history.append(guidance_scale_t)
            
            if debug and (i % 100 == 0 or i == len(timesteps) - 1):
                with torch.no_grad():
                    # Re-calculate x0_pred for logging if it wasn't computed in the guidance step
                    if not apply_guidance_this_step:
                         alpha_prod_t = self.scheduler.alphas_cumprod[t]
                         beta_prod_t = 1 - alpha_prod_t
                         z0_pred_local = (z_local - beta_prod_t.sqrt() * noise_pred_local) / alpha_prod_t.sqrt()
                         x0_pred = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z0_pred_local])
                    log_and_visualize_debug_info(self, i, len(timesteps), t.item(), z_local, z_global, noise_pred_local, current_loss, x0_pred)

        # --- 4. Final Processing & Decoding ---
        if debug and is_guidance_active:
            fig = plot_guidance_schedule(self, timesteps.cpu().numpy(), loss_history, scale_history)
            output_dict['guidance_schedule_plot'] = fig
            plt.close(fig)

        output_dict['z_global'] = z_global
        output_dict['z_local'] = z_local
        
        with torch.no_grad():
            # Final decoding to get the point cloud from the final latents
            alpha_prod_t = self.scheduler.alphas_cumprod[t] # t is now the last timestep
            beta_prod_t = 1 - alpha_prod_t
            noise_pred_local = local_prior(x=z_local, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
            z0_pred_local = (z_local - beta_prod_t.sqrt() * noise_pred_local) / alpha_prod_t.sqrt()
            output = self.vae.sample(num_samples=num_samples, decomposed_eps=[z_global, z0_pred_local])

        output_dict['points'] = output.detach()
        return output_dict