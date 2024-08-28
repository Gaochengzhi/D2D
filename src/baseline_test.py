from run_env import Highway_env
from gaodb import gaodb

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import gymnasium as gym
from itertools import product
import concurrent.futures

# Define parameter combinations
param_guess = {"a": [1], "b": [100], "c": [0.1, 1, 10]}


class GetParams(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.train = True

    def _on_training_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return self.train

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # Access the underlying environment
        env = getattr(self.locals["env"], "envs", [self.locals["env"]])[0]
        info = {
            "step": env.time_step,
            "mean_speed": env.mean_speed / (env.time_step + 1),
            "reward": env.total_reward,
            "task_level": env.task_level,
            "navigation": env.navigation_precent,
        }
        wandb.log(info)
        if env.reset_times >= 2000:
            self.train = False


def run_one_try(params):
    a, b, c = params

    # gaodb.init(project="merge", task="dive10")
    env = Highway_env(param=[a, b, c], gui=True)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000 * 30 * 400)
    # wandb.finish()


combinations = list(product(param_guess["a"], param_guess["b"], param_guess["c"]))


def parallel_run():
    # We use max_workers=22 to use 22 CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_one_try, params) for params in combinations]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will re-raise exceptions if they occurred
            except Exception as e:
                print(f"Exception occurred: {e}")


if __name__ == "__main__":
    # parallel_run()
    run_one_try([1, 100, 0.1])
