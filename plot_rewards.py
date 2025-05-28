import os
import json
import numpy as np
import matplotlib.pyplot as plt

envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]
algos = ["REINFORCE", "MiniBatchREINFORCE", "PPO", "A2C", "TRPO", "ARS"]
results_dir = "results"
max_episodes = 500

def load_rewards(env, algo):
    path = os.path.join(results_dir, algo, env)
    seeds = []
    if not os.path.exists(path):
        return []
    for fname in os.listdir(path):
        with open(os.path.join(path, fname), "r") as f:
            r = json.load(f)
            seeds.append(r[:max_episodes])
    return seeds

def smooth(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def plot_env_rewards(env):
    plt.figure(figsize=(10, 6))
    for algo in algos:
        runs = load_rewards(env, algo)
        if not runs:
            continue
        runs = [r[:max_episodes] for r in runs]
        min_len = min(len(r) for r in runs)
        runs = [r[:min_len] for r in runs]
        smoothed = np.array([smooth(np.array(r)) for r in runs])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        episodes = np.arange(len(mean))
        plt.plot(episodes, mean, label=algo)
        plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)

    plt.title(f"{env} - Smoothed Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{env.replace('-', '_')}_rewards.png")
    plt.close()

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    for env in envs:
        plot_env_rewards(env)
