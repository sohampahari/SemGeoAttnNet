import sys
import torch
import torch.nn as nn

# Update this path if needed
sys.path.insert(0, '/media/bigdata/71ec9ff9-bdc2-410a-bfcb-1a3e24aaf8f7/soham/3d-saliency/Pointcept')
from pointcept.models.point_transformer_v3 import PointTransformerV3

class SemGeoAttentionNet(nn.Module):
    def __init__(self, sem_dim=2048, hidden_dim=64):
        super().__init__()
        
        # 1. Geometry Stream (Point Transformer V3)
        self.geo_backbone = PointTransformerV3(
            in_channels=6,      # XYZ + Normals
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2),   # 3 elements for 3 decoder stages
            enc_depths=(2, 2, 6, 2),
            dec_depths=(1, 1, 1),  # Must be len(enc_depths)-1 = 3
            enc_channels=(32, 64, 128, 256),
            dec_channels=(64, 128, 256),  # Must be len(enc_depths)-1 = 3
            enc_num_head=(2, 4, 8, 16),
            dec_num_head=(4, 8, 16),  # Must be len(enc_depths)-1 = 3
            enc_patch_size=(48, 48, 48, 48),
            dec_patch_size=(48, 48, 48),  # Must be len(enc_depths)-1 = 3
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=False,  # flash_attn not available in environment
            upcast_attention=False,
            upcast_softmax=False
        )

        # 2. Geometry Projection
        # Backbone outputs dec_channels[0] = 64 channels
        backbone_out_dim = 64
        self.geo_proj = nn.Linear(backbone_out_dim, hidden_dim)
        self.geo_bn = nn.BatchNorm1d(hidden_dim)
        self.geo_act = nn.ReLU()

        # 3. Semantic Stream (Project 2048 -> hidden_dim)
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 4. Fusion (Cross-Attention)
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        # 5. Readout Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, coords, normals, semantics, batch_indices=None):
        B, N, _ = coords.shape
        
        # --- A. Geometry Path ---
        if batch_indices is None:
            batch_indices = torch.arange(B, device=coords.device).repeat_interleave(N)
            
        flat_coords = coords.reshape(-1, 3)
        flat_normals = normals.reshape(-1, 3)
        flat_feats = torch.cat([flat_coords, flat_normals], dim=-1)
        
        offset = torch.tensor([N] * B, device=coords.device).cumsum(0).int()
        
        data_dict = {
            "coord": flat_coords, 
            "feat": flat_feats, 
            "grid_size": 0.01, 
            "batch": batch_indices,
            "offset": offset
        }
        
        # Backbone Forward
        out_dict = self.geo_backbone(data_dict)
        geo_raw = out_dict['feat'] # Shape: (B*N, 64)
        
        # Project 64 -> hidden_dim
        geo_emb = self.geo_proj(geo_raw)
        geo_emb = self.geo_bn(geo_emb)
        geo_emb = self.geo_act(geo_emb)
        
        # Reshape to (B, N, C)
        geo_emb = geo_emb.view(B, N, -1)

        # --- B. Semantic Path ---
        sem_flat = semantics.reshape(-1, semantics.shape[-1])
        sem_emb = self.sem_proj(sem_flat) 
        sem_emb = sem_emb.view(B, N, -1)

        # --- C. Fusion ---
        attn_out, _ = self.fusion_attn(
            query=geo_emb, 
            key=sem_emb, 
            value=sem_emb
        )
        fused = geo_emb + attn_out

        # --- D. Prediction ---
        saliency = self.head(fused).squeeze(-1)
        
        return saliency