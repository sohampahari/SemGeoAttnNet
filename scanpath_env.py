import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
from scipy.spatial import KDTree

class ImprovedSal3DScanpathEnv(gym.Env):
    def __init__(self, data_dir="data", reward_dir="data/rl_rewards"):
        super().__init__()
        
        self.data_dir = data_dir
        self.reward_dir = reward_dir
        self.mesh_files = [f.replace('.pt', '') for f in os.listdir(reward_dir)]
        
        self.K = 6  # Number of neighbors
        self.action_space = spaces.Discrete(self.K)
        
        # Enhanced Observation Space:
        # [Current Saliency, IOR, neighbors_saliency (6), 
        #  neighbor_distances (6), progress (1), center_distance (1)]
        obs_dim = 2 + self.K + self.K + 1 + 1
        self.observation_space = spaces.Box(
            low=-1, high=10, shape=(obs_dim,), dtype=np.float32
        )
        
        self.current_mesh = None
        self.saliency_map = None
        self.kdtree = None
        self.current_idx = 0
        self.visited_counts = None
        self.steps = 0
        self.max_steps = 20
        self.trajectory = []
        
        # For normalization
        self.max_distance = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Load Random Mesh
        mesh_name = np.random.choice(self.mesh_files)
        self.saliency_map = torch.load(
            os.path.join(self.reward_dir, f"{mesh_name}.pt")
        ).numpy()
        
        # Load Geometry
        geo_path = os.path.join(self.data_dir, "geometry_data", f"{mesh_name}.pt")
        geo_data = torch.load(geo_path).numpy()
        self.vertices = geo_data[:, :3]
        
        # Build KDTree
        self.kdtree = KDTree(self.vertices)
        
        # Calculate mesh center and max distance for normalization
        self.mesh_center = np.mean(self.vertices, axis=0)
        self.max_distance = np.max(
            np.linalg.norm(self.vertices - self.mesh_center, axis=1)
        )
        
        # Start from center (more realistic for human viewing)
        center_distances = np.linalg.norm(
            self.vertices - self.mesh_center, axis=1
        )
        self.current_idx = np.argmin(center_distances)
        
        self.visited_counts = np.zeros(len(self.vertices))
        self.steps = 0
        self.trajectory = [self.current_idx]
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Find neighbors with distances
        dists, neighbor_idxs = self.kdtree.query(
            self.vertices[self.current_idx], k=self.K+1
        )
        self.neighbor_idxs = neighbor_idxs[1:]
        neighbor_dists = dists[1:]
        
        # Current state
        current_sal = self.saliency_map[self.current_idx]
        
        # Smooth IOR based on visit count (diminishing returns)
        current_ior = np.tanh(self.visited_counts[self.current_idx] / 3.0)
        
        # Neighbor features
        neigh_sal = self.saliency_map[self.neighbor_idxs]
        neigh_dists_norm = neighbor_dists / (self.max_distance + 1e-6)
        
        # Progress indicator
        progress = self.steps / self.max_steps
        
        # Distance from center (normalized)
        center_dist = np.linalg.norm(
            self.vertices[self.current_idx] - self.mesh_center
        ) / (self.max_distance + 1e-6)
        
        obs = np.concatenate([
            [current_sal, current_ior],
            neigh_sal,
            neigh_dists_norm,
            [progress, center_dist]
        ])
        
        return obs.astype(np.float32)

    def step(self, action):
        prev_idx = self.current_idx
        self.current_idx = self.neighbor_idxs[action]
        self.trajectory.append(self.current_idx)
        
        # ===== Improved Reward Function =====
        
        # 1. Saliency Reward (main signal)
        saliency_reward = float(self.saliency_map[self.current_idx])
        
        # 2. Progressive IOR Penalty (gets worse with repeated visits)
        visit_count = self.visited_counts[self.current_idx]
        ior_penalty = -0.2 * np.tanh(visit_count / 2.0)
        
        # 3. Exploration Bonus (encourage seeing new areas early)
        exploration_bonus = 0.0
        if visit_count == 0 and self.steps < self.max_steps * 0.7:
            exploration_bonus = 0.15
        
        # 4. Diversity Bonus (reward spreading out fixations)
        diversity_bonus = 0.0
        if len(self.trajectory) >= 3:
            # Check if recent fixations are spatially diverse
            recent_positions = self.vertices[self.trajectory[-3:]]
            pairwise_dists = np.linalg.norm(
                recent_positions[:-1] - recent_positions[1:], axis=1
            )
            if np.mean(pairwise_dists) > self.max_distance * 0.1:
                diversity_bonus = 0.1
        
        # 5. Small step penalty to encourage efficiency
        step_penalty = -0.05
        
        # Combine rewards
        reward = float(
            saliency_reward + 
            ior_penalty + 
            exploration_bonus + 
            diversity_bonus + 
            step_penalty
        )
        
        # Update visit counts with decay
        self.visited_counts[prev_idx] += 1.0
        
        self.steps += 1
        terminated = self.steps >= self.max_steps
        truncated = False
        
        info = {
            'saliency': saliency_reward,
            'ior_penalty': ior_penalty,
            'exploration': exploration_bonus,
            'diversity': diversity_bonus,
            'trajectory_length': len(self.trajectory)
        }
        
        return self._get_obs(), reward, terminated, truncated, info