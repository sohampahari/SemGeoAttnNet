import os
import torch
import numpy as np
import trimesh
from scipy.spatial import KDTree
from tqdm import tqdm

def align_gt_to_mesh(
    mesh_dir="meshes",                  # Folder with original .obj files
    gt_dir="ground_truth_dense",        # Folder with 20k .pt files
    output_dir="ground_truth_aligned"   # New output folder
):
    os.makedirs(output_dir, exist_ok=True)
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj')]
    
    print(f"Aligning {len(mesh_files)} meshes...")
    
    for mesh_file in tqdm(mesh_files):
        name = mesh_file.replace('.obj', '')
        gt_path = os.path.join(gt_dir, f"{name}.pt")
        
        if not os.path.exists(gt_path):
            continue
            
        # 1. Load Original Mesh (Target: ~50k)
        mesh = trimesh.load(os.path.join(mesh_dir, mesh_file), process=False)
        target_verts = mesh.vertices
        
        # 2. Load Sparse GT (Source: 20k)
        # We need the COORDINATES of the 20k GT to map them. 
        # Assuming you have the original SAL3D .txt files in 'gaze/'
        gaze_path = os.path.join("gaze", f"{name}.txt")
        if not os.path.exists(gaze_path):
            print(f"Missing gaze file for {name}")
            continue
            
        gaze_data = np.loadtxt(gaze_path)[:, :3] # Get XYZ of 20k points
        gt_vals = torch.load(gt_path).numpy()    # Get Saliency of 20k points
        
        # 3. KDTree Map (20k -> 50k)
        # Find which 20k point is closest to each 50k mesh vertex
        tree = KDTree(gaze_data)
        dists, idxs = tree.query(target_verts)
        
        # 4. Map Values
        aligned_gt = gt_vals[idxs]
        
        # 5. Save
        torch.save(torch.tensor(aligned_gt).float(), os.path.join(output_dir, f"{name}.pt"))
        
if __name__ == "__main__":
    align_gt_to_mesh()