import os
import torch
import numpy as np
import open3d as o3d
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
import pandas as pd

def calculate_chamfer_distance(pcd1_path, pcd2_path, device='cuda'):
    """
    Calculates the Chamfer Distance between two point clouds given their file paths.
    """
    try:
        # Load point clouds
        pcd1 = o3d.io.read_point_cloud(pcd1_path)
        pcd2 = o3d.io.read_point_cloud(pcd2_path)

        # Convert to torch tensors
        points1 = torch.from_numpy(np.asarray(pcd1.points)).float().unsqueeze(0)
        points2 = torch.from_numpy(np.asarray(pcd2.points)).float().unsqueeze(0)

        points1 = points1.to(device)
        points2 = points2.to(device)

        # Calculate Chamfer Distance
        chamfer_dist = chamfer_3DDist()
        dist1, dist2, _, _ = chamfer_dist(points1, points2)
        cd_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        
        return cd_loss.item()
    except Exception as e:
        print(f"Could not process files in {os.path.dirname(pcd1_path)}: {e}")
        return None

def analyze_results(log_dir='log', top_n=10):
    """
    Analyzes all experiment results in the log directory, calculates Chamfer Distance
    for each, and prints the top N best results.
    """
    results = []
    
    print(f"Analyzing experiments in '{log_dir}'...")

    # Walk through all subdirectories of the log directory
    for root, dirs, files in os.walk(log_dir):
        # Check if the directory contains the required result files
        if "01_gt.ply" in files and "03_recon.ply" in files:
            gt_path = os.path.join(root, "01_gt.ply")
            recon_path = os.path.join(root, "03_recon.ply")
            
            # The experiment name is the directory containing the time-stamped folder
            exp_path = root
            
            cd = calculate_chamfer_distance(gt_path, recon_path)
            
            if cd is not None:
                results.append({'experiment_path': exp_path, 'chamfer_distance': cd})

    if not results:
        print("No valid experiment results found.")
        return

    # Create a DataFrame and sort by Chamfer Distance
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='chamfer_distance', ascending=True)

    # Print the top N results
    print(f"\n--- Top {top_n} Experiments by Chamfer Distance ---")
    print(df_sorted.head(top_n).to_string())
    print("-------------------------------------------------")


if __name__ == '__main__':
    # Ensure CUDA is available if possible, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    analyze_results()
