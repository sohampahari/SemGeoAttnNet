import torch
from torch.utils.data import Dataset
import os
import numpy as np

class Sal3D_Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=2048):
        self.geo_dir = os.path.join(root_dir, "geometry_data")
        self.sem_dir = os.path.join(root_dir, "meshes_semantic")
        self.gt_dir = os.path.join(root_dir, "ground_truth_aligned")
        self.num_points = num_points
        
        # Get intersection of files present in all 3 folders
        geo_files = set(f.replace('.pt', '') for f in os.listdir(self.geo_dir))
        self.file_list = list(geo_files)
        
        # Simple split (80/20) - In prod, use a fixed list
        split_idx = int(len(self.file_list) * 0.8)
        if split == 'train':
            self.file_list = self.file_list[:split_idx]
        else:
            self.file_list = self.file_list[split_idx:]
            
        print(f"[{split.upper()}] Loaded {len(self.file_list)} meshes.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        
        # 1. Load Data
        # Geo: (N, 6) -> XYZ (3) + Normals (3)
        geo = torch.load(os.path.join(self.geo_dir, f"{fname}.pt")).float()
        
        # Sem: (N, 2048) -> Semantic Embeddings
        # Load as float16 (storage) -> convert to float32 (compute)
        sem = torch.load(os.path.join(self.sem_dir, f"{fname}.pt")).float()
        
        # GT: (M,) -> Saliency Map
        gt = torch.load(os.path.join(self.gt_dir, f"{fname}.pt")).float()
        
        # 2. Shape Safety Check
        # If N_geo != N_gt, we assume GT is valid and we must sample geometry 
        # (Assuming perfect alignment isn't guaranteed, we rely on random sampling)
        N = geo.shape[0]
        
        # 3. Sampling (FPS or Random)
        # For efficiency on H100, Random Sampling is often sufficient for dense tasks
        # Farthest Point Sampling (FPS) is better but slower.
        indices = torch.randperm(N)[:self.num_points]
        
        coord = geo[indices, :3]   # XYZ
        normal = geo[indices, 3:]  # Normals
        semantic = sem[indices, :] # 2048-dim features
        
        # Handle GT Sampling
        # If sizes match, sample GT using same indices. 
        # If sizes mismatch (e.g. Geo 50k vs GT 20k), we cannot trivially sample GT.
        # CRITICAL: We assume training relies on the input points. 
        # We return the FULL GT for loss interpolation.
        
        return {
            "coord": coord,         # (2048, 3)
            "normal": normal,       # (2048, 3)
            "semantic": semantic,   # (2048, 2048)
            "full_coord": geo[:,:3],# (N, 3) - Needed for Interpolation
            "full_gt": gt,          # (M,) - Target
            "name": fname
        }

def collate_fn(batch):
    """
    Custom collator to handle variable size 'full_coord' and 'full_gt' 
    while stacking the fixed-size training data.
    """
    coords = torch.stack([b['coord'] for b in batch])
    normals = torch.stack([b['normal'] for b in batch])
    semantics = torch.stack([b['semantic'] for b in batch])
    
    # Pack variable length data as lists
    full_coords = [b['full_coord'] for b in batch]
    full_gts = [b['full_gt'] for b in batch]
    names = [b['name'] for b in batch]
    
    return coords, normals, semantics, full_coords, full_gts, names