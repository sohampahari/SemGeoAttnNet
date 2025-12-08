from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os

# Local environment
from scanpath_env import ImprovedSal3DScanpathEnv

def train_improved_rl():
    # Create raw envs and run checks on the raw env (recommended)
    raw_train_env = ImprovedSal3DScanpathEnv()
    raw_eval_env = ImprovedSal3DScanpathEnv()

    # Validate the raw environment API
    check_env(raw_train_env)

    # Wrap with Monitor for logging when passing to SB3
    train_env = Monitor(raw_train_env)
    eval_env = Monitor(raw_eval_env)
    
    # Improved PPO configuration
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=1e-4,  # Lower learning rate
        n_steps=2048,  # Keep default
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_scanpath_tensorboard/"
    )
    
    # Ensure output folders exist
    os.makedirs('./models/best_model/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    os.makedirs('./models/checkpoints/', exist_ok=True)
    os.makedirs('./ppo_scanpath_tensorboard/', exist_ok=True)

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best_model/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/checkpoints/',
        name_prefix='scanpath_ppo'
    )
    
    # Train with more timesteps
    print("Starting Improved RL Training...")
    model.learn(
        total_timesteps=200000,  # More training
        callback=[eval_callback, checkpoint_callback]
    )
    
    model.save("scanpath_agent_ppo_improved")
    print("Agent Saved.")

if __name__ == "__main__":
    train_improved_rl()