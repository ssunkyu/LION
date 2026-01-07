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

import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist, SoftChamferDistance
from third_party.PyTorchEMD.emd import earth_mover_distance

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

def prepare_text_conditioning(model_cfg: dict, prompt:str, device: str):
    """Prepares text embedding (clip_feat) or a dummy tensor based on the prompt."""
    if prompt and prompt.strip():
        logging.info(f"Using text conditioning with prompt: '{prompt}'")
        clip_model, _ = clip.load(model_cfg.clipforge.clip_model, device=device)
        text = clip.tokenize([prompt]).to(device)
        clip_feat = clip_model.encode_text(text).float()
    else:
        logging.info("Using unconditional sampling (prompt is empty).")
        clip_feat = torch.zeros(1, model_cfg.clipforge.feat_dim, device=device)
        
    return clip_feat

def prepare_text_conditioning_old(cfg: dict, model_cfg: dict, device: str):
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

def load_data(data_cfg: dict, device: str):
    """
    Load a pcd data.
    """
    pcd_data_path = data_cfg['root']
    category = data_cfg['category_to_find']
    n_points = data_cfg['model_n_points']

    logging.info(f"Loading point cloud data from: {pcd_data_path}")
    logging.info(f"Category: '{category}', Number of samples: {n_points}")

    synset_id = find_synset_id(category, os.path.join(pcd_data_path, 'shapenet_synset_list.txt'))
    if not synset_id:
        logging.error(f"'{category}' cannot be found.")
        return None

    try:
        pcd_dir = os.path.join(pcd_data_path, synset_id, 'train')
        # pcd_filename = random.choice([f for f in os.listdir(pcd_dir) if f.endswith('.npy')])
        pcd_filename = os.listdir(pcd_dir)[0]
        pcd_path = os.path.join(pcd_dir, pcd_filename)
        logging.info(f"Ground Truth (x0) path: {pcd_path}")
        points = np.load(pcd_path)  # shape: (15000, 3)

    except (FileNotFoundError, IndexError) as e:
        logging.error(f"데이터 로드 실패: {pcd_dir}: {e}")
        return None

    # Random subsampling
    if points.shape[0] < n_points:
        # If < n_points, perform random sampling with replacement
        indices = np.random.choice(points.shape[0], n_points, replace=True)
    else:
        # If sufficient, perform random sampling without replacement
        indices = np.random.permutation(points.shape[0])[:n_points]
    
    points = points[indices, :] # shape: (n_points, 3) = (2048, 3)

    # Normalize pcd
    center = (np.amax(points, axis=0) + np.amin(points, axis=0)) / 2
    points = points - center

    scale = np.amax(np.linalg.norm(points, axis=1))
    if scale > 1e-8:
        points = points / scale

    x0 = torch.from_numpy(points).float().unsqueeze(0).to(device)

    x0 = rotate_pcd(x0, ry=90, device=device)

    logging.info(f"Complete data loading. Final shape: {x0.shape}")
    return x0

def rotate_pcd(pcd_tensor: torch.Tensor, rx: float = 0, ry: float = 0, rz: float = 0, device: str = 'cpu') -> torch.Tensor:
    """
    Applies rotation to a point cloud tensor.

    Args:
        pcd_tensor (torch.Tensor): The point cloud tensor of shape (B, N, 3).
        rx (float): Rotation angle in degrees around the X-axis.
        ry (float): Rotation angle in degrees around the Y-axis.
        rz (float): Rotation angle in degrees around the Z-axis.
        device (str): The device to perform computation on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The rotated point cloud tensor.
    """
    # 각도를 라디안으로 변환
    angle_x = torch.deg2rad(torch.tensor(rx, device=device))
    angle_y = torch.deg2rad(torch.tensor(ry, device=device))
    angle_z = torch.deg2rad(torch.tensor(rz, device=device))
    
    # X축 회전 행렬
    rotation_matrix_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_x), -torch.sin(angle_x)],
        [0, torch.sin(angle_x), torch.cos(angle_x)]
    ], device=device, dtype=pcd_tensor.dtype)

    # Y축 회전 행렬
    rotation_matrix_y = torch.tensor([
        [torch.cos(angle_y), 0, torch.sin(angle_y)],
        [0, 1, 0],
        [-torch.sin(angle_y), 0, torch.cos(angle_y)]
    ], device=device, dtype=pcd_tensor.dtype)

    # Z축 회전 행렬
    rotation_matrix_z = torch.tensor([
        [torch.cos(angle_z), -torch.sin(angle_z), 0],
        [torch.sin(angle_z), torch.cos(angle_z), 0],
        [0, 0, 1]
    ], device=device, dtype=pcd_tensor.dtype)

    # 전체 회전 행렬 계산 (Y, X, Z 순으로 회전 적용)
    rotation_matrix = rotation_matrix_y @ rotation_matrix_x @ rotation_matrix_z
    
    # 포인트 클라우드에 회전 적용 (행렬 곱)
    # pcd_tensor shape: (B, N, 3), rotation_matrix shape: (3, 3)
    rotated_pcd = torch.matmul(pcd_tensor, rotation_matrix)
    
    return rotated_pcd

def log_and_visualize_debug_info(self, i, total_steps, t, z_local, z_global, noise_pred_local, loss, x0_pred):
    """
    Logs debugging information and saves an intermediate visualization.
    """
    z_local_norm = torch.linalg.norm(z_local).item()
    z_local_std = torch.std(z_local).item()
    x0_pred_std = torch.std(x0_pred).item()
    x0_pred_abs_max = torch.max(torch.abs(x0_pred)).item()

    logging.info(
        f"\nDebug [Step {i:04d}/{total_steps}]"
        f" | CD Error: {loss:.4f}"
        f" | z_local Norm: {z_local_norm:.4f}"
        f" | z_local Std: {z_local_std:.4f}"
        f" | x0_pred Std: {x0_pred_std:.4f}"
        f" | x0_pred Max: {x0_pred_abs_max:.4f}"
    )
    
    # Save intermediate reconstruction
    plot_points(x0_pred, f"./tmp/recon_step_{i:04d}.png")

def plot_guidance_schedule(self, timesteps, loss_history, scale_history):
    """
    Generates and saves a plot of loss and guidance scale vs. timestep.
    Returns the figure object.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 1. Plot Loss on the left Y-axis (log scale)
    ax1.set_xlabel("Timestep (t)")
    ax1.set_ylabel("Loss (log scale)", color='tab:blue')
    ax1.plot(timesteps, loss_history, color='tab:blue', label='Loss', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", color='gray', alpha=0.5)

    # 2. Plot Guidance Scale on the right Y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Guidance Scale", color='tab:red')
    ax2.plot(timesteps, scale_history, color='tab:red', linestyle='--', label='Guidance Scale')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 3. Finalize plot
    plt.title("Guidance Loss & Scale vs. Timestep")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    fig.tight_layout()
    plt.savefig("./tmp/loss_and_scale_schedule.png")
    
    return fig

def load_data_older(cfg: dict, device: str):
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
        x0 = rotate_pcd(x0, ry=90, device=device)
        logging.info(f"Loaded and pre-processed ground truth data. Shape: {x0.shape}")
        return x0
        
    except (FileNotFoundError, IndexError) as e:
        logging.error(f"Could not load point cloud data from {pcd_dir}: {e}")
        return None

def load_data_random_older(cfg: dict, device: str):
    """
    DataLoader 클래스 없이, 단일 파일을 직접 로드하여
    정규화와 무작위 서브샘플링을 수행하는 함수.
    """
    pcd_data_path = cfg['paths']['pcd_data']
    category = cfg['experiment']['category_to_find']
    n_points = cfg['hyperparameters']['model_n_points']

    logging.info(f"단일 파일 로딩: 카테고리 '{category}', 샘플링 개수: {n_points}")

    # --- 1단계: 원본처럼 단일 .npy 파일 경로 찾기 ---
    # 참고: find_synset_id 함수가 utils에 정의되어 있다고 가정합니다.
    synset_id = find_synset_id(category, os.path.join(pcd_data_path, 'shapenet_synset_list.txt'))
    if not synset_id:
        logging.error(f"'{category}' 카테고리를 찾을 수 없습니다.")
        return None

    try:
        pcd_dir = os.path.join(pcd_data_path, synset_id, 'train')
        # 매번 같은 파일을 로드하지 않도록 무작위로 하나의 파일 선택
        # pcd_filename = random.choice([f for f in os.listdir(pcd_dir) if f.endswith('.npy')])
        pcd_filename = os.listdir(pcd_dir)[0]
        pcd_path = os.path.join(pcd_dir, pcd_filename)
        logging.info(f"Ground Truth (x0) 로드 경로: {pcd_path}")
        points = np.load(pcd_path)  # shape: (15000, 3)

    except (FileNotFoundError, IndexError) as e:
        logging.error(f"데이터 로드 실패: {pcd_dir}: {e}")
        return None

    # --- 2단계: 무작위 서브샘플링 수행 ---
    if points.shape[0] < n_points:
        # 포인트 개수가 부족하면 복원 추출
        indices = np.random.choice(points.shape[0], n_points, replace=True)
    else:
        # 충분하면 비복원 추출
        indices = np.random.permutation(points.shape[0])[:n_points]
    
    points = points[indices, :] # shape: (n_points, 3)

    # --- 3단계: 포인트 클라우드 정규화 ([-1, 1] 범위로) ---
    # 포인트 클라우드의 중심을 원점(0,0,0)으로 이동
    center = (np.amax(points, axis=0) + np.amin(points, axis=0)) / 2
    points = points - center

    # 최대 거리가 1이 되도록 스케일 조정
    scale = np.amax(np.linalg.norm(points, axis=1))
    if scale > 1e-8:  # 0으로 나누는 것 방지
        points = points / scale

    # --- 4단계: 텐서 변환 및 후처리 ---
    # 배치 차원(1) 추가 후 디바이스로 이동
    x0 = torch.from_numpy(points).float().unsqueeze(0).to(device)

    # 이전과 동일하게 회전 변환 적용
    x0 = rotate_pcd(x0, ry=90, device=device)

    logging.info(f"데이터 로드 및 처리 완료. 최종 Shape: {x0.shape}")
    return x0