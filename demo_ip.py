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
from functools import partial
import debugpy

from models.lion import LION
from data.dataloader import get_dataset, get_dataloader
from utils.measurements import get_operator, get_noise
from utils.condition_methods import get_conditioning_method
from default_config import cfg as model_config
from utils.demo_utils import (
    setup_logging,
    save_point_cloud,
    prepare_text_conditioning,
    load_data
)

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser(description="Run LION inverse problem solver with custom hyperparameters.")
    parser.add_argument('--model_config', type=str, default='./config/all_uncond_cfg.yml', help="Path to the base config file.")
    parser.add_argument('--task_config', type=str, default='./config/ip_cfg.yml')
    
    parser.add_argument('--pcd_data', type=str, default='./data/ShapeNetCore.v2.PC15k')
    # parser.add_argument('--model_ckpt', type=str, default='./lion_ckpt/text2shape/chair/checkpoints/model.pt')
    # parser.add_argument('--model_ckpt', type=str, default='./lion_ckpt/unconditional/chair/checkpoints/model.pt')
    parser.add_argument('--model_ckpt', type=str, default='./lion_ckpt/unconditional/all55/checkpoints/epoch_10999_iters_2100999.pt')
    # parser.add_argument('--model_ckpt', type=str, default='./lion_ckpt/unconditional/all55/samples.pt')
    parser.add_argument('--log_dir', type=str, default='./log')

    parser.add_argument('--cond_method', type=str, default='ps_linear', help="Type of guidance (e.g., 'dps', 'dps_linear', '').")
    parser.add_argument('--guidance_start', type=float, help="Guidance scale at start.")
    parser.add_argument('--guidance_end', type=float, help="Guidance scale at end.")

    parser.add_argument('--debug', action='store_true', help="Enable debugging.")
    args = parser.parse_args()
    
    model_config.merge_from_file(args.model_config)
    task_config = load_yaml(args.task_config)

    if args.cond_method is None:
        task_config['conditioning']['method'] = args.cond_method
    if args.guidance_start is not None:
        task_config['conditioning']['params']['guidance_start'] = args.guidance_start
    if args.guidance_end is not None:
        task_config['conditioning']['params']['guidance_end'] = args.guidance_end
    
    if args.debug:
        logging.info("Debug mode is enabled.")
        debugpy.listen(5678)
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")
    
    time_str = datetime.now().strftime('%H%M%S')

    output_dir = setup_logging(args.log_dir, time_str)
    
    # Save the current configuration to the output directory
    with open(os.path.join(output_dir, 'model_config.yml'), 'w') as f:
        yaml.dump(model_config, f)
        f.close()
    with open(os.path.join(output_dir, 'task_config.yml'), 'w') as f:
        yaml.dump(task_config, f)
        f.close()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Load model
    model = LION(model_config)
    model.load_model(args.model_ckpt)
    model.to(device)
    
    operator_config = task_config['measurement']['operator']
    operator = get_operator(**operator_config, device=device)

    noise_config = task_config['measurement']['noise']
    noiser = get_noise(**noise_config)
    
    clip_feat = prepare_text_conditioning(model_config, task_config['prompt'], device)
    
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning

    # Load data
    data_config = task_config['data']
    x = load_data(data_config, device)
    
    if x is None:
        logging.error("Failed to load data. Aborting.")
        return

    # assert x.shape[0] == clip_feat.shape[0], "Batch size of data and text conditioning must match."

    # Run DPS sampling
    y = operator.forward(x)
    y = noiser.forward(y)
    logging.info("Running DPS sampling...")
    
    output = model.dps_sample(
        measurement=y,
        measurement_cond_fn=measurement_cond_fn,
        clip_feat=clip_feat,
    )

    # 6. Save results
    logging.info("Saving results...")
    pts = output['points']
    
    save_point_cloud(x, os.path.join(output_dir, "01_gt"))
    save_point_cloud(y, os.path.join(output_dir, "02_measure"))
    save_point_cloud(pts, os.path.join(output_dir, "03_recon"))
    
    logging.info(f"Script finished successfully. Results saved in: {output_dir}")

if __name__ == '__main__':
    main()
