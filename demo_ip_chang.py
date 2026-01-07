# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
    Requires diffusers-0.11.1
    This script demonstrates guided point cloud generation using DPS.
    1. Loads a ground truth point cloud from ShapeNet.
    2. Creates a degraded measurement 'y' (e.g., down-sampled).
    3. Runs DPS to generate a point cloud that fits both a text prompt and the measurement 'y'.
"""
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# ... (copyright header remains the same) ...

import os
import torch
import yaml
import shutil
import logging
import argparse
import json
from datetime import datetime

from models.lion_chang import LION
from utils import measurements
from default_config import cfg as model_config
from utils.demo_utils import (
    setup_logging,
    save_point_cloud,
    prepare_text_conditioning_old,
    get_loss_function,
    load_data_random_older,
)

def run(cfg: dict, model_cfg: dict, args: argparse.Namespace):
    """Main execution function for the guided generation."""
    time_str = datetime.now().strftime('%H%M%S')
    op_name = cfg['experiment']['forward_operator']
    loss_names = '_'.join(l['name'] for l in cfg['experiment']['loss_functions'])
    hp = cfg['hyperparameters']
    
    exp_name = (
        f"op-{op_name}_loss-{loss_names}_"
        f"gs{hp['guidance_scale']}_cs{hp['consistency_scale']}_"
        f"t{hp['guidance_start_t']}-{hp['guidance_stop_t']}_"
        f"sch-{hp['guidance_scheduler']}"
    )
    
    output_dir = setup_logging(cfg['paths']['log_dir'], exp_name, time_str)
    
    # Save the current configuration to the output directory
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(cfg, f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # Log the main guidance hyperparameters
    logging.info("--- Guidance Hyperparameters ---")
    logging.info(f"Guidance scale: {cfg['hyperparameters']['guidance_scale']}")
    logging.info(f"Guidance start time: {cfg['hyperparameters']['guidance_start_t']}")
    logging.info(f"Scheduler: {cfg['hyperparameters']['guidance_scheduler']}")
    logging.info("---------------------------------")


    # 2. Load model
    lion = LION(model_config)
    lion.load_model(cfg['paths']['model'])
    lion.to(device)

    # 3. Prepare utilities: loss function, forward model, and conditioning
    loss_fn = get_loss_function(cfg)
    forward_model = measurements.get_operator(name=cfg['experiment']['forward_operator'], device=device)
    clip_feat = prepare_text_conditioning_old(cfg, model_cfg, device)

    # 4. Load data
    x0 = load_data_random_older(cfg, device)
    if x0 is None:
        logging.error("Failed to load data. Aborting.")
        return

    # 5. Run DPS sampling
    y_measurement = forward_model.forward(x0)
    logging.info("Running DPS sampling...")
    
    output = lion.dps_sample(
        y=y_measurement,
        loss_fn=loss_fn,
        guidance_start_t=cfg['hyperparameters']['guidance_start_t'],
        guidance_stop_t=cfg['hyperparameters']['guidance_stop_t'],
        guidance_scale=cfg['hyperparameters']['guidance_scale'],
        guidance_scheduler=cfg['hyperparameters']['guidance_scheduler'],
        num_samples=clip_feat.shape[0],
        forward_model=forward_model,
        clip_feat=clip_feat,
        debug=cfg['debug_mode']
    )
    
    # output = lion.plsd_sample(
    #     y=y_measurement,
    #     loss_fn=loss_fn,
    #     guidance_start_t=cfg['hyperparameters']['guidance_start_t'],
    #     guidance_stop_t=cfg['hyperparameters']['guidance_stop_t'],
    #     guidance_scale=cfg['hyperparameters']['guidance_scale'],
    #     consistency_scale=cfg['hyperparameters']['consistency_scale'],
    #     guidance_scheduler=cfg['hyperparameters']['guidance_scheduler'],
    #     num_samples=clip_feat.shape[0],
    #     forward_model=forward_model,
    #     clip_feat=clip_feat,
    #     debug=cfg['debug_mode']
    # )

    # 6. Save results
    logging.info("Saving results...")
    pts = output['points']
    
    save_point_cloud(x0, os.path.join(output_dir, "01_gt"))
    save_point_cloud(y_measurement, os.path.join(output_dir, "02_measure"))
    save_point_cloud(pts, os.path.join(output_dir, "03_recon"))
    
    logging.info(f"Script finished successfully. Results saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run LION inverse problem solver with custom hyperparameters.")
    parser.add_argument('--config', type=str, default='./config/ip_cfg_chang.yml', help="Path to the base config file.")
    parser.add_argument('--guidance_scale', type=float, help="Guidance scale.")
    parser.add_argument('--guidance_start_t', type=int, help="Guidance start timestep.")
    parser.add_argument('--guidance_scheduler', type=str, help="Guidance scheduler (e.g., 'linear', 'constant').")
    parser.add_argument('--loss_functions', type=str, help="JSON string of loss functions and weights, e.g., '[{\"name\": \"chamfer\", \"weight\": 1.0}]'")
    parser.add_argument('--debug_mode', type=lambda x: (str(x).lower() == 'true'), default=None, help="Enable or disable debug mode.")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_main = yaml.safe_load(f)

    # Override config with command-line arguments if provided
    if args.guidance_scale is not None:
        config_main['hyperparameters']['guidance_scale'] = args.guidance_scale
    if args.guidance_start_t is not None:
        config_main['hyperparameters']['guidance_start_t'] = args.guidance_start_t
    if args.guidance_scheduler is not None:
        config_main['hyperparameters']['guidance_scheduler'] = args.guidance_scheduler
    if args.loss_functions is not None:
        config_main['experiment']['loss_functions'] = json.loads(args.loss_functions)
    if args.debug_mode is not None:
        config_main['hyperparameters']['debug_mode'] = args.debug_mode

    model_config.merge_from_file(config_main['paths']['model_cfg'])
    
    run(config_main, model_config, args)

if __name__ == '__main__':
    main()
