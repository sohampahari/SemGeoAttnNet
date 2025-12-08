import os
import torch
import numpy as np
import trimesh
from scipy.spatial import KDTree
from tqdm import tqdm

def align_gt_to_mesh(
    mesh_dir="meshes",                  # Folder containing .obj files
    gaze_dir="gaze",                    # Folder containing .txt files (Source coords)
    gt_dir="ground_truth_dense",        # Folder containing 20k .pt files (Source values)
    output_dir="data/ground_truth_aligned"   # DESTINATION folder
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all meshes
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj')]
    print(f"Aligning {len(mesh_files)} meshes...")
    
    for mesh_file in tqdm(mesh_files):
        name = mesh_file.replace('.obj', '')
        
        # 1. Load the Target Mesh (High Res, ~50k)
        # This defines the "correct" number of vertices
        mesh_path = os.path.join(mesh_dir, mesh_file)
        try:
            mesh = trimesh.load(mesh_path, process=False)
            target_verts = mesh.vertices # Shape: (N_high, 3)
        except:
            print(f"Skipping {name}: Could not load mesh.")
            continue

        # 2. Load the Source Coordinates (Low Res, 20k)
        # We need to know WHERE the 20k GT points are located 3D space
        gaze_txt_path = os.path.join(gaze_dir, f"{name}.txt")
        if not os.path.exists(gaze_txt_path):
            print(f"Skipping {name}: No raw gaze file found in {gaze_dir}")
            continue
            
        try:
            # Load XYZ coordinates (first 3 columns)
            source_coords = np.loadtxt(gaze_txt_path)[:, :3] 
        except:
            print(f"Skipping {name}: Error reading text file.")
            continue

        # 3. Load the Source Values (Low Res, 20k)
        gt_pt_path = os.path.join(gt_dir, f"{name}.pt")
        if not os.path.exists(gt_pt_path):
            print(f"Skipping {name}: No GT .pt file found.")
            continue
            
        gt_values = torch.load(gt_pt_path, weights_only=True).numpy()

        # Sanity Check: Coords and Values must match length
        if len(source_coords) != len(gt_values):
            # If mismatch, try to truncate the larger one (rare edge case in SAL3D)
            min_len = min(len(source_coords), len(gt_values))
            source_coords = source_coords[:min_len]
            gt_values = gt_values[:min_len]

        # 4. Map Low-Res GT -> High-Res Mesh using KDTree
        # For every point in the High-Res mesh, find the nearest neighbor in the Low-Res GT
        tree = KDTree(source_coords)
        dists, idxs = tree.query(target_verts, k=1) # indices of nearest source point
        
        # Create the new High-Res GT
        aligned_gt = gt_values[idxs]
        
        # Optional: Zero out points that are too far from any GT point (outliers)
        # aligned_gt[dists > 0.05] = 0.0 
        
        # 5. Save
        # The result shape is now (N_high,), matching the mesh!
        save_path = os.path.join(output_dir, f"{name}.pt")
        torch.save(torch.tensor(aligned_gt).float(), save_path)

if __name__ == "__main__":
    # Ensure these paths match your actual folder names!
    align_gt_to_mesh(
        mesh_dir="meshes", 
        gaze_dir="gaze", 
        gt_dir="ground_truth_dense",
        output_dir="data/ground_truth_aligned"
    )