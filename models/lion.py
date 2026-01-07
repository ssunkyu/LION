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
               measurement: torch.Tensor,
               measurement_cond_fn,
               clip_feat,
               ):
        """
        Performs guided sampling using Diffusion Policy Gradient (DPS).
        """
        # --- 1. 초기 설정 ---
        n_timesteps = 1000
        self.scheduler.set_timesteps(n_timesteps, device=measurement.device)
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]

        # history-tracking lists
        sampled_list = []
        output_dict = {}

        # start sample global prior
        z_shape = [measurement.shape[0]] + latent_shape[0]
        z_noisy = torch.randn(size=z_shape, device=measurement.device)
        # condition_input = None
        with torch.no_grad():
            for t in tqdm(timesteps, desc="Sampling Global Prior"):
                t_tensor = torch.full((measurement.shape[0],), t + 1, device=measurement.device, dtype=torch.long)
                z_noise_pred = global_prior(x=z_noisy, t=t_tensor.float(), clip_feat=clip_feat)
                z_noisy = self.scheduler.step(z_noise_pred, t, z_noisy).prev_sample

        z0 = z_noisy
        sampled_list.append(z0)
        output_dict['z_global'] = z0

        condition_input = z0
        condition_input = self.vae.global2style(condition_input)

        # start sample local prior
        h_shape = [measurement.shape[0]] + latent_shape[1]
        h_noisy = torch.randn(size=h_shape, device=measurement.device)

        pbar = tqdm(timesteps, desc="Sampling Local Prior")
        for i, t in enumerate(pbar):
            t_tensor = torch.full((measurement.shape[0],), t + 1, device=measurement.device, dtype=torch.long)

            with torch.enable_grad():
                h_noisy = h_noisy.detach().requires_grad_(True)

                h_noise_pred = local_prior(x=h_noisy, t=t_tensor.float(), condition_input=condition_input, clip_feat=clip_feat)
                h_noisy_t = self.scheduler.step(h_noise_pred, t, h_noisy).prev_sample
                
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                h0_pred = (h_noisy - beta_prod_t.sqrt() * h_noise_pred) / alpha_prod_t.sqrt()
                x0_pred = self.vae.sample(num_samples=measurement.shape[0], decomposed_eps=[z0, h0_pred])
                progress = i / len(timesteps)
                
                h_noisy, distance = measurement_cond_fn(x_prev=h_noisy,
                                                    x_t=h_noisy_t,
                                                    x_0_hat=x0_pred,
                                                    measurement=measurement,
                                                    progress=progress)
                
                current_dist = distance.item() if isinstance(distance, torch.Tensor) else distance
                pbar.set_postfix({'distance': current_dist}, refresh=False)
        
        h0 = h_noisy
        x0 = self.vae.sample(num_samples=measurement.shape[0], decomposed_eps=[z0, h0])
        output_dict['points'] = x0.detach()

        return output_dict



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def dps_sample_old(self,
               y: torch.Tensor,
               loss_fn: torch.nn.Module,
               measurement_cond_fn,
               num_samples: int = 1,
               clip_feat=None,
               ):
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
                guidance_scale_t = 20 - 0.02 * t.item()
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

    def get_mixing_component(self, noise_pred, t):
        # usage:
        # if global_prior.mixed_prediction:
        #     mixing_component = self.get_mixing_component(noise_pred, t)
        #     coeff = torch.sigmoid(global_prior.mixing_logit)
        #     noise_pred = (1 - coeff) * mixing_component + coeff * noise_pred

        alpha_bar = self.scheduler.alphas_cumprod[t]
        one_minus_alpha_bars_sqrt = np.sqrt(1.0 - alpha_bar)
        return noise_pred * one_minus_alpha_bars_sqrt
