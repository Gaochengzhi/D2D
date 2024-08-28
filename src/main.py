import os
import gymnasium as gym
import numpy as np
from run_env import Highway_env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from gaodb import gaodb

# 创建保存模型和日志的目录

total_episodes = 500
save_freq = 100

gaodb.init(project="merge", task="sac10")


# 训练和评估函数
def train():
    # Wandb初始化
    env = Highway_env(gui=False)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
    )
    model.learn(total_timesteps=70000)


train()
