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
import os
import clip
import torch
import numpy as np
from PIL import Image
import logging
from datetime import datetime

from default_config import cfg as config
from models.lion import LION
from utils.vis_helper import plot_points
from huggingface_hub import hf_hub_download
from utils import measurements
from utils.utils import find_synset_id
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DFunction, chamfer_3DDist

# 1. Setup Paths, Configs, and Hyperparameters
pcd_data_path = './data/ShapeNetCore.v2.PC15k'
model_path = './lion_ckpt/text2shape/chair/checkpoints/model.pt'
model_config = './lion_ckpt/text2shape/chair/cfg.yml'
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters for this run
category_to_find = 'chair'  # You can change this to 'table', 'car', etc.
sampling_method = 'dps'
forward_operator= 'denoise_pcd'
guidance_scale = 0.5  # Tune this hyperparameter
debug_mode = False
MODEL_N_POINTS = 2048 # LION VAE가 출력하는 포인트 개수

# Create a unique directory for this run's outputs
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
output_dir_name = f"{timestamp}_{sampling_method}_{forward_operator}_gs-{guidance_scale}_{category_to_find}"
output_dir = os.path.join('./log', output_dir_name)
os.makedirs(output_dir, exist_ok=True)

# Setup logging to console and a file inside the output directory
# --- Setup logging to console and a file inside the output directory ---
log_file_path = os.path.join(output_dir, 'run.log')

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear any existing handlers to avoid duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

# Create a formatter
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Create a file handler to write to the log file
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Create a stream handler to print to the console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# 2. Load Model
logging.info("Loading LION model...")
config.merge_from_file(model_config)
lion = LION(config)
lion.load_model(model_path)
lion.to(device_str)
# lion.vae.half()
# lion.priors.half()

# 3. Load Forward Model (A) and Loss Function
logging.info("Loading Forward Model (A) from measurements module...")
# forward_model = measurements.get_operator(name=forward_operator, num_points_out=1024, device=device_str)
# forward_model = measurements.get_operator(name=forward_operator, k=32, sigma=0.1, device=device_str)
forward_model = measurements.get_operator(name=forward_operator, device=device_str)
loss_fn = chamfer_3DDist()

# 4. Prepare Text Condition (CLIP Features)

if config.clipforge.enable:
    input_t = ["a comfortable leather armchair"]
    clip_model, _ = clip.load(config.clipforge.clip_model, device=device_str)
    text = clip.tokenize(input_t).to(device_str)
    clip_feat = []
    clip_feat.append(clip_model.encode_text(text).float())
    clip_feat = torch.cat(clip_feat, dim=0)
    logging.info(f"CLIP feature shape: {clip_feat.shape}")
else:
    clip_feat = None

# 5. Load Ground Truth Point Cloud (x0)
metadata_path = os.path.join(pcd_data_path, 'shapenet_synset_list.txt')
chair_synset_id = find_synset_id(category_to_find, metadata_path)

x0 = None
if chair_synset_id:
    chair_pcd_dir = os.path.join(pcd_data_path, chair_synset_id, 'train')
    try:
        chair_files = os.listdir(chair_pcd_dir)
        if not chair_files:
            raise FileNotFoundError(f"No files found in {chair_pcd_dir}")
        
        pcd_filename = chair_files[0]
        pcd_path = os.path.join(chair_pcd_dir, pcd_filename)
        logging.info(f"Loading Ground Truth (x0) from: {pcd_path}")
        
        x0 = torch.from_numpy(np.load(pcd_path)).float().to(device_str)
        
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)
        
        num_samples = 1 if clip_feat is None else clip_feat.shape[0]
        if x0.shape[0] != num_samples:
            x0 = x0.repeat(num_samples, 1, 1)

    except FileNotFoundError:
        logging.error(f"Could not find ShapeNet data at the specified path: {chair_pcd_dir}")
else:
    logging.error(f"Could not find synset ID for category '{category_to_find}'.")
    
# --- 5.1. Pre-process x0 to match the model's output point count ---
x0_matched = None
if x0 is not None:
    logging.info(f"Original x0 has {x0.shape[1]} points. Downsampling to {MODEL_N_POINTS} to match model output size.")
    
    # Use the super_resolution operator (which is FPS) to downsample x0
    downsampler = measurements.get_operator(name='super_resolution_pcd', num_points_out=MODEL_N_POINTS, device=device_str)
    x0 = downsampler.forward(x0)
    
    logging.info(f"Downsampled (size-matched) x0 shape: {x0.shape}")

# 6. Create Measurement (y) and Run DPS Sampling 
if x0 is not None:
    logging.info(f"Creating measurement (y = A(x0)) with original shape {x0.shape} -> downsample")
    y_measurement = forward_model.forward(x0)
    logging.info(f"y_measurement shape: {y_measurement.shape}")
    
    logging.info("Running DPS sampling...")
    output = lion.dps_sample(y=y_measurement,
                             loss_fn=loss_fn,
                             guidance_scale=guidance_scale,
                             num_samples=num_samples,
                             forward_model=forward_model,
                             clip_feat=clip_feat,
                             debug=debug_mode)

    # 7. Visualize and Save All Results 
    logging.info(" Analyzing and Saving result images ")
    pts = output['points']
    
    # START FINAL OUTPUT DEBUGGING 
    logging.info(f"Final output 'pts' shape: {pts.shape}")
    logging.info(f"Contains NaN: {torch.isnan(pts).any().item()}")
    logging.info(f"Contains Inf: {torch.isinf(pts).any().item()}")
    logging.info(f"Min value: {pts.min().item():.4f}, Max value: {pts.max().item():.4f}")
    # END FINAL OUTPUT DEBUGGING 

    # Save the ground truth image
    gt_path = os.path.join(output_dir, "01_ground_truth_x0.png")
    plot_points(x0, output_name=gt_path)
    logging.info(f"Saved ground truth image: {gt_path}")

    # Save the measurement image
    measurement_path = os.path.join(output_dir, "02_measurement_y.png")
    plot_points(y_measurement, output_name=measurement_path)
    logging.info(f"Saved measurement image: {measurement_path}")
    
    # Save the final reconstructed result
    recon_path = os.path.join(output_dir, f"03_reconstructed_output.png")
    plot_points(pts, output_name=recon_path)
    logging.info(f"Saved final result: {recon_path}")
    
    logging.info(f"Script finished successfully. Check the directory: {output_dir}")