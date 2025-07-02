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
import clip
import torch
import numpy as np
import logging
import yaml # Import the YAML library
from datetime import datetime
import shutil

from default_config import cfg as config
from models.lion import LION
from utils.vis_helper import plot_points
from utils import measurements
from utils.utils import find_synset_id
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

# ==========================================================================================
# 1. Load Configuration from YAML file
# ==========================================================================================
config_path = './config/ip_cfg.yml'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

# --- Extract parameters from the loaded config ---
paths_cfg = cfg['paths']
exp_cfg = cfg['experiment']
hyper_cfg = cfg['hyperparameters']

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================================================================
# 2. Setup Logging and Output Directory
# ==========================================================================================
timestamp = datetime.now().strftime('%y%m%d_%H%M')
output_dir_name = timestamp
output_dir = os.path.join(paths_cfg['log_dir'], output_dir_name)
os.makedirs(output_dir, exist_ok=True)

shutil.copy(config_path, output_dir)
logging.info(f"Copied config file to {output_dir}")

# --- Setup logging to console and file ---
log_file_path = os.path.join(output_dir, 'run.log')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


# ==========================================================================================
# 3. Load Model
# ==========================================================================================
logging.info("Loading LION model...")
config.merge_from_file(paths_cfg['model_cfg'])
lion = LION(config)
lion.load_model(paths_cfg['model'])
lion.to(device_str)


# ==========================================================================================
# 4. Load Forward Model (A) and Loss Function
# ==========================================================================================
logging.info(f"Loading Forward Model (A): {exp_cfg['forward_operator']}")
forward_model = measurements.get_operator(name=exp_cfg['forward_operator'], device=device_str)
loss_fn = chamfer_3DDist()


# ==========================================================================================
# 5. Prepare Text Condition (CLIP Features)
# ==========================================================================================
if config.clipforge.enable:
    input_t = [exp_cfg['prompt']]
    clip_model, _ = clip.load(config.clipforge.clip_model, device=device_str)
    text = clip.tokenize(input_t).to(device_str)
    clip_feat = clip_model.encode_text(text).float()
    logging.info(f"CLIP feature shape: {clip_feat.shape}")
else:
    clip_feat = None
num_samples = 1 if clip_feat is None else clip_feat.shape[0]


# ==========================================================================================
# 6. Load Ground Truth Point Cloud (x0)
# ==========================================================================================
metadata_path = os.path.join(paths_cfg['pcd_data'], 'shapenet_synset_list.txt')
synset_id = find_synset_id(exp_cfg['category_to_find'], metadata_path)

x0 = None
if synset_id:
    pcd_dir = os.path.join(paths_cfg['pcd_data'], synset_id, 'train')
    try:
        pcd_filename = os.listdir(pcd_dir)[0]
        pcd_path = os.path.join(pcd_dir, pcd_filename)
        logging.info(f"Loading Ground Truth (x0) from: {pcd_path}")
        
        x0 = torch.from_numpy(np.load(pcd_path)).float().to(device_str)
        x0 = x0.unsqueeze(0) if x0.dim() == 2 else x0
        x0 = x0.repeat(num_samples, 1, 1) if x0.shape[0] != num_samples else x0

    except (FileNotFoundError, IndexError):
        logging.error(f"Could not find or load data from the specified path: {pcd_dir}")
else:
    logging.error(f"Could not find synset ID for category '{exp_cfg['category_to_find']}'.")

# --- Pre-process x0 to match the model's output point count ---
if x0 is not None:
    model_n_points = hyper_cfg['model_n_points']
    logging.info(f"Original x0 has {x0.shape[1]} points. Matching to {model_n_points} points.")
    downsampler = measurements.get_operator(name='super_resolution_pcd', num_points_out=model_n_points, device=device_str)
    x0 = downsampler.forward(x0)
    logging.info(f"Downsampled (size-matched) x0 shape: {x0.shape}")


# ==========================================================================================
# 7. Create Measurement (y) and Run DPS Sampling
# ==========================================================================================
if x0 is not None:
    logging.info(f"Creating measurement y = A(x0)")
    y_measurement = forward_model.forward(x0)
    logging.info(f"y_measurement shape: {y_measurement.shape}")
    
    logging.info("Running DPS sampling...")
    output = lion.dps_sample(
        y=y_measurement,
        loss_fn=loss_fn,
        guidance_scale=hyper_cfg['guidance_scale'],
        guidance_start_t=hyper_cfg['guidance_start_t'],
        num_samples=num_samples,
        forward_model=forward_model,
        clip_feat=clip_feat,
        debug=hyper_cfg['debug_mode']
    )

    # ======================================================================================
    # 8. Visualize and Save All Results
    # ======================================================================================
    logging.info("Analyzing and Saving result images.")
    pts = output['points']
    
    # --- Final Output Sanity Check ---
    logging.info(f"Final output 'pts' shape: {pts.shape}")
    logging.info(f"Contains NaN: {torch.isnan(pts).any().item()}")
    logging.info(f"Contains Inf: {torch.isinf(pts).any().item()}")

    # --- Save the point clouds as images ---
    gt_path = os.path.join(output_dir, "01_ground_truth_x0.png")
    plot_points(x0, output_name=gt_path)

    measurement_path = os.path.join(output_dir, "02_measurement_y.png")
    plot_points(y_measurement, output_name=measurement_path)
    
    recon_path = os.path.join(output_dir, "03_reconstructed_output.png")
    plot_points(pts, output_name=recon_path)
    
    logging.info(f"Saved results to: {recon_path}")
    logging.info(f"Script finished successfully. Check the directory: {output_dir}")