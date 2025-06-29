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
import logging # Import the logging module

from default_config import cfg as config
from models.lion import LION
from utils.vis_helper import plot_points
from huggingface_hub import hf_hub_download
# Assuming the PCD operators are saved in utils/measurements.py
from utils import measurements
# Assuming the find_synset_id function is saved in utils/utils.py
from utils.utils import find_synset_id
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DFunction, chamfer_3DDist

# --- Setup basic logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 1. Setup Paths and Configs ---
pcd_data_path = './data/ShapeNetCore.v2.PC15k'
model_path = './lion_ckpt/text2shape/chair/checkpoints/model.pt'
model_config = './lion_ckpt/text2shape/chair/cfg.yml'
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
category_to_find = 'chair'  # You can change this to 'table', 'car', etc.

# --- 2. Load Model ---
logging.info("Loading LION model...")
config.merge_from_file(model_config)
lion = LION(config)
lion.load_model(model_path)
lion.to(device_str)

# --- 3. Load Forward Model (A) ---
logging.info("Loading Forward Model (A) from measurements module...")
# Using super-resolution as an example. This operator downsamples the point cloud.
# You can change this to other operators like 'denoise_pcd' or 'inpainting_pcd'.
forward_model = measurements.get_operator(name='super_resolution_pcd', num_points_out=10000, device=device_str)
loss_fn = chamfer_3DDist()

# --- 4. Prepare Text Condition (CLIP Features) ---
if config.clipforge.enable:
    input_t = ["a comfortable leather armchair"]
    clip_model, _ = clip.load(config.clipforge.clip_model, device=device_str)
    text = clip.tokenize(input_t).to(device_str)
    clip_feat = clip_model.encode_text(text).float()
    logging.info(f"CLIP feature shape: {clip_feat.shape}")
else:
    clip_feat = None

# --- 5. Load Ground Truth Point Cloud (x0) ---
# Find synset ID automatically
metadata_path = os.path.join(pcd_data_path, 'shapenet_synset_list.txt')
chair_synset_id = find_synset_id(category_to_find, metadata_path)

x0 = None
if chair_synset_id:
    # Assuming a standard train/test/val split in the dataset directory
    chair_pcd_dir = os.path.join(pcd_data_path, chair_synset_id, 'train')
    try:
        chair_files = os.listdir(chair_pcd_dir)
        if not chair_files:
            raise FileNotFoundError(f"No files found in {chair_pcd_dir}")
        
        # Use the first file in the directory as the ground truth
        pcd_filename = chair_files[0]
        pcd_path = os.path.join(chair_pcd_dir, pcd_filename)
        logging.info(f"Loading Ground Truth (x0) from: {pcd_path}")
        
        # Load .npy file and convert to a processed PyTorch tensor
        x0 = torch.from_numpy(np.load(pcd_path)).float().to(device_str)
        
        # Adjust tensor shape: (N, 3) -> (B, N, 3)
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)
        
        # Match the batch size with the text condition
        num_samples = 1 if clip_feat is None else clip_feat.shape[0]
        if x0.shape[0] != num_samples:
            x0 = x0.repeat(num_samples, 1, 1)

    except FileNotFoundError:
        logging.error(f"Could not find ShapeNet data at the specified path: {chair_pcd_dir}")
        logging.error("Please check your pcd_data_path and the directory structure.")
else:
    logging.error(f"Could not find synset ID for category '{category_to_find}'.")

# --- 6. Create Measurement (y) and Run DPS Sampling ---
if x0 is not None:
    # Create the measurement y by applying the forward model to x0
    logging.info(f"Creating measurement (y = A(x0)) with original shape {x0.shape} -> downsample")
    y_measurement = forward_model.forward(x0)
    logging.info(f"y_measurement shape: {y_measurement.shape}")
    
    # Run DPS sampling
    logging.info("Running DPS sampling...")
    guidance_scale = 0.5  # Guidance scale. Tune this hyperparameter.
    output = lion.dps_sample(y=y_measurement,
                             loss_fn=loss_fn,
                             guidance_scale=guidance_scale,
                             num_samples=num_samples,
                             forward_model=forward_model,
                             clip_feat=clip_feat)
    
    # --- 7. Visualize and Save All Results ---
    logging.info("Saving result images to ./tmp/ directory...")
    
    # Save the ground truth image
    plot_points(x0, output_name="./tmp/01_ground_truth_x0.png")
    logging.info("Saved ground truth image: ./tmp/01_ground_truth_x0.png")

    # Save the measurement image
    plot_points(y_measurement, output_name="./tmp/02_measurement_y.png")
    logging.info("Saved measurement image: ./tmp/02_measurement_y.png")
    
    # Save the final reconstructed result
    pts = output['points']
    img_name = f"./tmp/03_reconstructed_output_gs_{guidance_scale}.png"
    plot_points(pts, output_name=img_name)
    logging.info(f"Saved final result: {img_name}")
    
    logging.info("Script finished. Check the ./tmp/ directory for the 3 output images.")