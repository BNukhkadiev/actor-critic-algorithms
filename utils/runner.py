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

    if issubclass(algo_cls, BaseAlgorithm):
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)

        agent = algo_cls("MlpPolicy", env, seed=seed, verbose=0)
        agent.learn(total_timesteps=total_timesteps)

        # Extract reward per episode with associated timesteps
        rewards = env.get_episode_rewards()
        lengths = env.get_episode_lengths()

        timestep_rewards = []
        cumulative_timestep = 0
        for r, l in zip(rewards, lengths):
            cumulative_timestep += l
            timestep_rewards.append((cumulative_timestep, r))

    else:
        # Custom algorithm should return list of (timestep, reward) tuples
        agent = algo_cls(env_id, seed)
        timestep_rewards = agent.train()

    with open(output_file, "w") as f:
        json.dump(timestep_rewards, f)
