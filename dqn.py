import os
import gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import SpaceShooterEnv  

# Learning rate schedule function
def lr_schedule(progress_remaining):
    return 1e-4 * progress_remaining

# Create environment
env = DummyVecEnv([lambda: SpaceShooterEnv()])

# Check if CUDA is available and select device
device = 'cuda'
print(f"Using device: {device}")

# Model file path
model_file = "ppo_space_shooter_env.zip"

# Check if the model file exists
if os.path.exists(model_file):
    print("Loading the saved model...")
    # Load the existing model with weights_only=True for security
    model = PPO.load(model_file, env=env, device=device, weights_only=True)
else:
    print("No saved model found. Creating a new model...")
    # Create a new model if no saved model exists
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=lr_schedule, n_steps=2048, batch_size=64, device=device)
    # Train the model
    model.learn(total_timesteps=10000)
    # Save the model after training
    model.save(model_file)

# Run episodes using the model
for episode in range(100):
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action, _ = model.predict(state)
        # Take the action in the environment
        next_state, reward, done, _ = env.step(action)
        # Accumulate the reward
        episode_reward += reward
        env.render(mode='None')
    
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
