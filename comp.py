import trimesh
import matplotlib.pyplot as plt
import os

# Path to your results
viz_dir = "results_viz_normalized"
files = [f for f in os.listdir(viz_dir) if f.endswith("_pred.ply")]

# Plot first 3 results
fig, axes = plt.subplots(3, 2, figsize=(10, 15))

for i, fname in enumerate(files[:3]):
    # Load Pred and GT
    mesh_pred = trimesh.load(os.path.join(viz_dir, fname))
    mesh_gt = trimesh.load(os.path.join(viz_dir, fname.replace("_pred", "_gt")))
    
    # Get colors
    colors_pred = mesh_pred.visual.vertex_colors[:, :3] / 255.0
    colors_gt = mesh_gt.visual.vertex_colors[:, :3] / 255.0
    
    # Project to 2D (XY plane view)
    pts = mesh_pred.vertices
    
    axes[i, 0].scatter(pts[:, 0], pts[:, 1], c=colors_pred, s=1)
    axes[i, 0].set_title(f"{fname} (Prediction)")
    axes[i, 0].axis('off')
    
    axes[i, 1].scatter(pts[:, 0], pts[:, 1], c=colors_gt, s=1)
    axes[i, 1].set_title(f"Ground Truth")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig("comparison_view.png")
print("Saved comparison_view.png")