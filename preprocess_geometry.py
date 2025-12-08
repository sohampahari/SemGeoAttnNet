import os
import torch
import trimesh
import numpy as np
from tqdm import tqdm

def preprocess_geometry(
    mesh_dir="meshes/",
    output_dir="geometry_data/"
):
    """
    Reads .obj files and saves (Positions + Normals) as .pt tensors.
    Output Shape: (N, 6) -> [x, y, z, nx, ny, nz]
    """
    os.makedirs(output_dir, exist_ok=True)
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj')]
    
    print(f"Pre-processing geometry for {len(mesh_files)} meshes...")
    
    for mesh_file in tqdm(mesh_files):
        name = mesh_file.replace('.obj', '')
        
        # Load mesh (force distinct vertices to match other data)
        try:
            # process=False is CRITICAL to keep vertex order identical 
            # to your 'semantics' and 'smooth_gaze' data
            mesh = trimesh.load(os.path.join(mesh_dir, mesh_file), process=False)
        except Exception as e:
            print(f"Failed to load {mesh_file}: {e}")
            continue
            
        # 1. Get Vertices (N, 3)
        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        
        # 2. Get/Compute Normals (N, 3)
        # Some .obj files have vertex normals, some don't.
        try:
            # Try to read existing vertex normals
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals.shape == verts.shape:
                normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)
            else:
                # Fallback: Compute them (might be slightly different from original)
                mesh.fix_normals()
                normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)
        except:
             # Final fallback: Zeros (rare)
             normals = torch.zeros_like(verts)

        # 3. Concatenate (N, 6)
        geo_tensor = torch.cat([verts, normals], dim=1)
        
        # 4. Save
        torch.save(geo_tensor, os.path.join(output_dir, f"{name}.pt"))

if __name__ == "__main__":
    preprocess_geometry()