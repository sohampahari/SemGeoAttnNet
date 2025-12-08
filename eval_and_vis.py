import torch
import numpy as np
import os
import trimesh
from tqdm import tqdm
from dataset import Sal3D_Dataset, collate_fn
from model import SemGeoAttentionNet
import matplotlib.pyplot as plt

def apply_heatmap(saliency, cmap='jet'):
    """
    Min-Max normalizes the input to [0, 1] range before coloring.
    This ensures low-magnitude probability maps are visible.
    """
    # Convert to numpy if tensor
    if isinstance(saliency, torch.Tensor):
        saliency = saliency.cpu().numpy()
        
    # Min-Max Normalization (Per shape)
    v_min = saliency.min()
    v_max = saliency.max()
    
    # Avoid division by zero for flat maps
    if v_max - v_min > 1e-8:
        saliency = (saliency - v_min) / (v_max - v_min)
    
    cmap = plt.get_cmap(cmap)
    return cmap(saliency)[:, :3]

def interpolate_dense(pred, low_res_coords, high_res_coords):
    # Chunking to avoid OOM
    chunk_size = 5000
    N = high_res_coords.shape[0]
    interpolated = torch.zeros(N, device=low_res_coords.device)
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk_tgt = high_res_coords[i:end].unsqueeze(0) 
        chunk_src = low_res_coords.unsqueeze(0)        
        
        dists = torch.cdist(chunk_tgt, chunk_src).squeeze(0)
        vals, idxs = torch.topk(dists, k=3, dim=1, largest=False)
        weights = 1.0 / (vals + 1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        neighbor_vals = pred[idxs]
        interpolated[i:end] = (neighbor_vals * weights).sum(dim=1)
        
    return interpolated

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "results_viz_normalized" # New folder
    os.makedirs(out_dir, exist_ok=True)

    val_ds = Sal3D_Dataset("data", split='test') 
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    print(f"Re-visualizing {len(val_ds)} meshes with Normalization...")

    model = SemGeoAttentionNet(sem_dim=2048, hidden_dim=32).to(device)
    checkpoint = "checkpoint_ep100.pth" 
    
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
    else:
        print("Error: Checkpoint not found.")
        return

    model.eval()
    
    with torch.no_grad():
        for coords, normals, semantics, full_coords, full_gts, names in tqdm(val_loader):
            coords, normals, semantics = coords.to(device), normals.to(device), semantics.to(device)
            
            preds = model(coords, normals, semantics)
            pred_high_res = interpolate_dense(preds[0], coords[0], full_coords[0].to(device))
            gt_high_res = full_gts[0].to(device)
            
            # --- Visualization Saving ---
            # 1. Prediction (Normalized)
            colors_pred = apply_heatmap(pred_high_res)
            mesh_verts = full_coords[0].cpu().numpy()
            pcd_pred = trimesh.points.PointCloud(mesh_verts, colors=(colors_pred * 255).astype(np.uint8))
            pcd_pred.export(os.path.join(out_dir, f"{names[0]}_pred.ply"))
            
            # 2. Ground Truth (Normalized - THIS FIXES THE BLUE IMAGES)
            colors_gt = apply_heatmap(gt_high_res)
            pcd_gt = trimesh.points.PointCloud(mesh_verts, colors=(colors_gt * 255).astype(np.uint8))
            pcd_gt.export(os.path.join(out_dir, f"{names[0]}_gt.ply"))

    print("Done. Check 'results_viz_normalized/'")

if __name__ == "__main__":
    main()