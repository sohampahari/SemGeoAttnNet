import torch
from dataset import Sal3D_Dataset, collate_fn
from model import SemGeoAttentionNet
from tqdm import tqdm
import os

def cache_rewards(
    data_dir="data", 
    out_dir="data/rl_rewards", 
    checkpoint="checkpoint_ep100.pth"
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = SemGeoAttentionNet(sem_dim=2048, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    
    # Load Dataset (All files)
    # Note: We need a dataset mode that returns ALL files, not just train/test
    # For now, we assume split='train' + 'test' covers it, or create a custom list
    ds = Sal3D_Dataset(data_dir, split='train') 
    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    
    print("Caching Saliency Rewards...")
    with torch.no_grad():
        for coords, normals, semantics, full_coords, _, names in tqdm(loader):
            coords, normals, semantics = coords.to(device), normals.to(device), semantics.to(device)
            
            # Predict Saliency (0-1)
            preds = model(coords, normals, semantics) # (1, 2048)
            
            # Map to Full Mesh Resolution (The "World")
            # We use the interpolate function from previous steps
            from train import interpolate_to_gt
            full_saliency = interpolate_to_gt(preds, coords, full_coords)[0]
            
            # Save
            torch.save(full_saliency.cpu(), os.path.join(out_dir, f"{names[0]}.pt"))

if __name__ == "__main__":
    cache_rewards()