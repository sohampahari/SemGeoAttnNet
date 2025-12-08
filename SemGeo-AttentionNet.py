import sys
import torch
import torch.nn as nn

# Update path if necessary
sys.path.insert(0, '/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/soham/3d-saliency/Pointcept')
from pointcept.models.point_transformer_v3 import PointTransformerV3

class SemGeoAttentionNet(nn.Module):
    def __init__(self, sem_dim=2048, hidden_dim=64):
        super().__init__()
        
        # 1. Geometry Stream (Point Transformer V3)
        self.geo_backbone = PointTransformerV3(
            in_channels=6,      # XYZ + Normals
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            # FIX: Stride length must be len(enc_depths) - 1
            # 4 Stages defined below -> Requires 3 Strides
            stride=(2, 2, 2),
            enc_depths=(2, 2, 6, 2),
            dec_depths=(1, 1, 1),  # Must be len(enc_depths) - 1 = 3
            enc_channels=(32, 64, 128, 256),
            dec_channels=(64, 128, 256),  # Must be len(enc_depths) - 1 = 3
            enc_num_head=(2, 4, 8, 16),
            dec_num_head=(4, 8, 16),  # Must match dec_depths length
            enc_patch_size=(48, 48, 48, 48),
            dec_patch_size=(48, 48, 48),  # Must match dec_depths length
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=False,  # Disabled: flash_attn not installed. Set to True after: pip install flash-attn
            upcast_attention=False,
            upcast_softmax=False
        )

        # 2. Semantic Stream (Project 2048 -> 64 to match PTv3 output)
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 3. Fusion (Cross-Attention)
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        # 4. Readout Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Saliency is 0-1 probability
        )

    def forward(self, coords, normals, semantics, batch_indices=None):
        """
        coords: (B, N, 3)
        normals: (B, N, 3)
        semantics: (B, N, 2048)
        batch_indices: (B*N,) or None
        """
        B, N, _ = coords.shape
        
        # --- A. Geometry Path ---
        if batch_indices is None:
            # Create batch indices: [0,0...0, 1,1...1, ...]
            batch_indices = torch.arange(B, device=coords.device).repeat_interleave(N)
            
        flat_coords = coords.reshape(-1, 3)
        flat_normals = normals.reshape(-1, 3)
        flat_feats = torch.cat([flat_coords, flat_normals], dim=-1)  # (B*N, 6)
        
        # PTv3 Data Dict
        offset = torch.tensor([N] * B, device=coords.device).cumsum(0).int()
        
        data_dict = {
            "coord": flat_coords, 
            "feat": flat_feats, 
            "grid_size": 0.01, 
            "batch": batch_indices,
            "offset": offset
        }
        
        # Forward pass - output features in data_dict['feat']
        out_dict = self.geo_backbone(data_dict)
        geo_emb = out_dict['feat']  # Shape: (B*N, 32)
        
        # Reshape back to (B, N, C)
        geo_emb = geo_emb.view(B, N, -1)

        # --- B. Semantic Path ---
        sem_flat = semantics.reshape(-1, semantics.shape[-1])
        sem_emb = self.sem_proj(sem_flat)  # (B*N, 32)
        sem_emb = sem_emb.view(B, N, -1)

        # --- C. Fusion ---
        # Residual Cross-Attention
        attn_out, _ = self.fusion_attn(
            query=geo_emb, 
            key=sem_emb, 
            value=sem_emb
        )
        fused = geo_emb + attn_out

        # --- D. Prediction ---
        saliency = self.head(fused).squeeze(-1) # (B, N)
        
        return saliency

if __name__ == "__main__":
    # Sanity Check
    print("=" * 70)
    print("SemGeoAttentionNet - Sanity Check")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    
    print("\n📦 Instantiating model...")
    try:
        model = SemGeoAttentionNet(sem_dim=2048, hidden_dim=64).to(device)
        print("✅ Model instantiated successfully")
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n📊 Creating dummy input data...")
    B, N = 2, 2048
    coords = torch.rand(B, N, 3).to(device)
    normals = torch.rand(B, N, 3).to(device)
    semantics = torch.rand(B, N, 2048).to(device)
    
    print("\n🚀 Running forward pass...")
    try:
        output = model(coords, normals, semantics)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
        
        # Basic Output Validation
        if output.shape == (B, N):
             print("✅ Output shape matches expected (B, N)")
        else:
             print(f"❌ Output shape mismatch! Got {output.shape}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()