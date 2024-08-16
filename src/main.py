import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="human")

model = PPO("MlpPolicy", "CartPole-v1").learn(10_000)