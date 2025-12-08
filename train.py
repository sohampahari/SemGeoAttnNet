import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import os

# Ensure these files exist in your folder:
from dataset import Sal3D_Dataset, collate_fn
from model import SemGeoAttentionNet # Rename SemGeo-AttentionNet.py to model.py

def calc_metrics(pred, gt):
    """Calculates KL-Divergence and Correlation Coefficient"""
    eps = 1e-7
    
    # Normalize to probability distribution
    pred_norm = pred / (pred.sum() + eps)
    gt_norm = gt / (gt.sum() + eps)
    
    # 1. KL Divergence
    kld = torch.sum(gt_norm * torch.log(gt_norm / (pred_norm + eps) + eps))
    
    # 2. Correlation Coefficient
    pred_mean = pred - pred.mean()
    gt_mean = gt - gt.mean()
    numerator = torch.sum(pred_mean * gt_mean)
    denominator = torch.sqrt(torch.sum(pred_mean**2)) * torch.sqrt(torch.sum(gt_mean**2))
    cc = numerator / (denominator + eps)
    
    return kld, cc

def interpolate_to_gt(pred_saliency, network_coords, full_coords):
    """
    Maps 2048 predictions -> Full Mesh Resolution using Inverse Distance Weighting
    """
    interpolated_preds = []
    
    for i in range(len(full_coords)):
        src_pos = network_coords[i]       # (2048, 3)
        src_val = pred_saliency[i]        # (2048,)
        tgt_pos = full_coords[i].to(src_pos.device) # (N_full, 3)
        
        # Compute distances (N_full, 2048)
        # Note: If N_full is very large (>50k), this might consume memory.
        # Consider processing in chunks if you hit OOM.
        dists = torch.cdist(tgt_pos.unsqueeze(0), src_pos.unsqueeze(0)).squeeze(0)
        
        # 3 Nearest Neighbors
        vals, idxs = torch.topk(dists, k=3, dim=1, largest=False)
        
        # IDW Weights
        weights = 1.0 / (vals + 1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Weighted Sum
        neighbor_vals = src_val[idxs]
        interp = (neighbor_vals * weights).sum(dim=1)
        interpolated_preds.append(interp)
        
    return interpolated_preds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Configuration
    BATCH_SIZE = 8 # Safe for H100 with 50k points
    LR = 1e-4
    EPOCHS = 100
    DATA_DIR = "data" # Ensure your folders are inside here

    # Data
    # Make sure Sal3D_Dataset points to 'ground_truth_aligned' now!
    train_ds = Sal3D_Dataset(DATA_DIR, split='train')
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    
    print(f"Training on {len(train_ds)} meshes.")

    # Model
    model = SemGeoAttentionNet(sem_dim=2048, hidden_dim=32).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_kld = 0
        total_cc = 0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for coords, normals, semantics, full_coords, full_gts, names in pbar:
            coords, normals, semantics = coords.to(device), normals.to(device), semantics.to(device)
            
            optimizer.zero_grad()
            
            # 1. Predict (Low Res)
            preds = model(coords, normals, semantics) # (B, 2048)
            
            # 2. Interpolate (High Res)
            full_preds = interpolate_to_gt(preds, coords, full_coords)
            
            # 3. Compute Loss (High Res)
            loss = 0
            batch_kld_sum = 0
            batch_cc_sum = 0
            valid_samples = 0
            
            for i, (p_full, gt_full) in enumerate(zip(full_preds, full_gts)):
                gt_full = gt_full.to(device)
                
                # Mismatch Guard
                if p_full.shape[0] != gt_full.shape[0]:
                    print(f"Skipping {names[i]}: Shape mismatch Pred={p_full.shape} GT={gt_full.shape}")
                    continue
                
                kld, cc = calc_metrics(p_full, gt_full)
                
                # Hybrid Loss: Minimize KLD, Maximize CC
                # Weights: KLD is dominant (10x), CC is regularizer
                sample_loss = (10 * kld) - (2 * cc)
                loss += sample_loss
                
                batch_kld_sum += kld.item()
                batch_cc_sum += cc.item()
                valid_samples += 1
            
            if valid_samples > 0:
                loss = loss / valid_samples
                loss.backward()
                optimizer.step()
                
                avg_kld = batch_kld_sum / valid_samples
                avg_cc = batch_cc_sum / valid_samples
                pbar.set_postfix({'KLD': f"{avg_kld:.4f}", 'CC': f"{avg_cc:.4f}"})
                
                total_kld += avg_kld
                total_cc += avg_cc
                batch_count += 1

        if batch_count > 0:
            print(f"Epoch {epoch+1} Summary | Avg KLD: {total_kld/batch_count:.4f} | Avg CC: {total_cc/batch_count:.4f}")
        
        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_ep{epoch+1}.pth")

if __name__ == "__main__":
    main()