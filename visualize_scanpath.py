import torch
import numpy as np
import os
import trimesh
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from scanpath_env import ImprovedSal3DScanpathEnv # Assuming you saved your class in a file
import imageio

def generate_scanpath_vis(
    model_path="scanpath_agent_ppo_improved", 
    data_dir="data", 
    reward_dir="data/rl_rewards",
    output_dir="scanpath_results"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Environment & Model
    env = ImprovedSal3DScanpathEnv(data_dir=data_dir, reward_dir=reward_dir)
    model = PPO.load(model_path)
    
    # Run on 3 random meshes
    for i in range(3):
        obs, _ = env.reset()
        done = False
        
        mesh_name = env.mesh_files[env.mesh_files.index(os.path.basename(env.current_mesh) if env.current_mesh else np.random.choice(env.mesh_files))]
        # Note: The env logic picks random on reset, so we grab the name used
        # We need to hack into the env slightly to get the name if not stored:
        # Let's assume for visualization we just let it run.
        
        print(f"Generating scanpath for episode {i+1}...")
        
        trajectory_indices = [env.current_idx]
        total_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=False) # False = Natural human variance
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            trajectory_indices.append(env.current_idx)
            
        # Get Geometry for plotting
        vertices = env.vertices
        path_points = vertices[trajectory_indices]
        saliency = env.saliency_map
        
        # --- Visualization 1: Matplotlib 3D ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Point Cloud (Subsampled for speed)
        skip = 5 
        # Color by saliency
        p = ax.scatter(vertices[::skip, 0], vertices[::skip, 1], vertices[::skip, 2], 
                       c=saliency[::skip], cmap='viridis', s=1, alpha=0.3)
        
        # Plot Scanpath
        # Line
        ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 
                color='red', linewidth=2, label='Scanpath')
        
        # Fixation Points (Start = Green, End = Red)
        ax.scatter(path_points[0, 0], path_points[0, 1], path_points[0, 2], 
                   color='lime', s=100, label='Start', edgecolors='black')
        ax.scatter(path_points[-1, 0], path_points[-1, 1], path_points[-1, 2], 
                   color='red', s=100, label='End', edgecolors='black')
        
        # Plot numbered markers
        for step, point in enumerate(path_points):
            ax.text(point[0], point[1], point[2], str(step), fontsize=9, color='black')

        ax.set_title(f"Generated Scanpath (Reward: {total_reward:.2f})")
        ax.legend()
        plt.savefig(os.path.join(output_dir, f"scanpath_{i}.png"))
        plt.close()
        
        # --- Visualization 2: Export to PLY for MeshLab ---
        # We create a mesh of just the path (edges)
        import trimesh
        
        # Create a Path visual
        path_visual = trimesh.load_path(path_points)
        
        # Create Points visual with Saliency Colors
        # (This is tricky in pure trimesh, easier to save vertices + scalar field)
        # We save the underlying mesh with saliency colors
        norm_sal = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        colors = plt.get_cmap('viridis')(norm_sal)[:, :3] * 255
        mesh_cloud = trimesh.points.PointCloud(vertices, colors=colors.astype(np.uint8))
        
        mesh_cloud.export(os.path.join(output_dir, f"mesh_{i}.ply"))
        
        # Save trajectory as simple CSV for external plotting
        np.savetxt(os.path.join(output_dir, f"traj_{i}.csv"), path_points, delimiter=",")
        
    print(f"Done. Results saved to {output_dir}")

if __name__ == "__main__":
    generate_scanpath_vis()