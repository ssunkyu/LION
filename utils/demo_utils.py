import os
import clip
import torch
import numpy as np
import logging
import yaml
from datetime import datetime
import open3d as o3d

from utils.utils import find_synset_id
from utils.vis_helper import plot_points
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist, SoftChamferDistance
from third_party.PyTorchEMD.emd import earth_mover_distance
from utils import measurements

def setup_logging(log_dir: str, exp_name: str = "default", time_str: str = None):
    """Sets up the logging directory."""
    if time_str is None:
        time_str = datetime.now().strftime('%H%M%S')
    
    date_str = datetime.now().strftime('%y%m%d')
    
    # Create a directory structure: log/{date}/{exp_name}/{time}
    output_dir = os.path.join(log_dir, date_str, exp_name, time_str)
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    return output_dir

def save_point_cloud(pcd_tensor: torch.Tensor, filename_prefix: str):
    """Saves a point cloud to both .png and .ply files."""
    # Save as .png image
    plot_points(pcd_tensor, output_name=f"{filename_prefix}.png")
    
    # Save as .ply 3D file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_tensor.squeeze(0).cpu().numpy())
    o3d.io.write_point_cloud(f"{filename_prefix}.ply", pcd)

def prepare_text_conditioning(cfg: dict, model_cfg: dict, device: str):
    """Prepares text embedding (clip_feat) or a dummy tensor based on the prompt."""
    prompt = cfg['experiment'].get('prompt', "")
    
    if prompt and prompt.strip():
        logging.info(f"Using text conditioning with prompt: '{prompt}'")
        clip_model, _ = clip.load(model_cfg.clipforge.clip_model, device=device)
        text = clip.tokenize([prompt]).to(device)
        clip_feat = clip_model.encode_text(text).float()
    else:
        logging.info("Using unconditional sampling (prompt is empty).")
        clip_feat = torch.zeros(1, model_cfg.clipforge.feat_dim, device=device)
        
    return clip_feat

def get_loss_function(cfg: dict):
    """
    Creates a master loss function by combining multiple loss functions 
    specified in the config, with their respective weights.
    """
    loss_configs = cfg['experiment']['loss_functions']
    logging.info(f"Loading and combining loss functions: {loss_configs}")

    chamfer_dist = chamfer_3DDist()
    
    loss_calculators = {
        'chamfer': lambda p1, p2: torch.mean(chamfer_dist(p1, p2)[0]) + torch.mean(chamfer_dist(p1, p2)[1]),
        'emd': lambda p1, p2: torch.mean(earth_mover_distance(p1, p2, transpose=False)),
        'soft_chamfer': lambda p1, p2: SoftChamferDistance(temperature=cfg['hyperparameters']['soft_chamfer_temp'])(p1, p2)
    }

    active_losses = []
    for config in loss_configs:
        name = config['name']
        weight = config['weight']
        if name in loss_calculators:
            active_losses.append({'calc': loss_calculators[name], 'weight': weight})
        else:
            logging.warning(f"Loss function '{name}' is not defined. Skipping.")

    if not active_losses:
        raise ValueError("No valid loss functions were specified in the configuration.")

    def master_loss_fn(p1, p2):
        total_loss = 0.0
        for loss_info in active_losses:
            total_loss += loss_info['calc'](p1, p2) * loss_info['weight']
        return total_loss

    return master_loss_fn

def load_data(cfg: dict, device: str):
    """Loads and pre-processes the ground truth point cloud."""
    pcd_data_path = cfg['paths']['pcd_data']
    category = cfg['experiment']['category_to_find']
    
    synset_id = find_synset_id(category, os.path.join(pcd_data_path, 'shapenet_synset_list.txt'))
    if not synset_id:
        logging.error(f"Category '{category}' not found.")
        return None

    try:
        pcd_dir = os.path.join(pcd_data_path, synset_id, 'train')
        pcd_filename = os.listdir(pcd_dir)[0]
        pcd_path = os.path.join(pcd_dir, pcd_filename)
        
        logging.info(f"Loading Ground Truth (x0) from: {pcd_path}")
        x0 = torch.from_numpy(np.load(pcd_path)).float().to(device).unsqueeze(0)
    
        downsampler = measurements.get_operator(
            name='super_resolution_pcd', 
            num_points_out=cfg['hyperparameters']['model_n_points'], 
            device=device
        )
        x0 = downsampler.forward(x0)
        logging.info(f"Loaded and pre-processed ground truth data. Shape: {x0.shape}")
        return x0
        
    except (FileNotFoundError, IndexError) as e:
        logging.error(f"Could not load point cloud data from {pcd_dir}: {e}")
        return None
