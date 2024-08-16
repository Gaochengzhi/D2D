import gymnasium as gym
from run_env import Highway_env
from stable_baselines3 import PPO

env = Highway_env(gui=False)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_00_000)
