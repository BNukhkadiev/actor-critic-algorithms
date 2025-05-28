import os
import json
import gymnasium as gym
import numpy as np

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor

def run_experiment(env_id, algo_cls, algo_name, seed, total_timesteps=50_000):
    print(f"Running {algo_name} on {env_id} with seed {seed}")

    # Create result path
    result_path = f"results/{algo_name}/{env_id}"
    os.makedirs(result_path, exist_ok=True)
    output_file = os.path.join(result_path, f"seed{seed}.json")

    # SB3/SB3-contrib algorithms
    if issubclass(algo_cls, BaseAlgorithm):
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)

        agent = algo_cls("MlpPolicy", env, seed=seed, verbose=0)
        agent.learn(total_timesteps=total_timesteps)

        # Get episode rewards from Monitor
        rewards = env.get_episode_rewards()

    else:
        # Custom algorithm (assumed compatible interface)
        agent = algo_cls(env_id, seed)
        rewards = agent.train()

    with open(output_file, "w") as f:
        json.dump(rewards, f)
