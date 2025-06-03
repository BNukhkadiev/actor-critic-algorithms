import os
import json
import numpy as np
import matplotlib.pyplot as plt

envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]
# envs = ["CartPole-v1"]

# algos = ["A2C", "PPO", "TRPO"]
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
    all_rewards = []

    algo_to_runs = {}

    # Step 1: Load all runs for all algos and collect all raw rewards
    for algo in algos:
        runs = load_rewards(env, algo)  # each run: list of (timestep, reward) tuples
        if not runs:
            continue
        all_rewards.extend([r for run in runs for _, r in run])
        algo_to_runs[algo] = runs

    if not all_rewards:
        print(f"No data for {env}")
        return

    # Step 2: Normalize
    min_r = min(all_rewards)
    max_r = max(all_rewards)

    for algo, runs in algo_to_runs.items():
        normalized_runs = []
        for run in runs:
            normalized_run = [(t, (r - min_r) / (max_r - min_r + 1e-8)) for t, r in run]
            normalized_runs.append(normalized_run)

        # Interpolate to fixed timestep grid (e.g., every 100 timesteps)
        x_vals = np.linspace(0, 50000, 300)
        interpolated = []
        for run in normalized_runs:
            timesteps, rewards = zip(*run)
            interpolated.append(np.interp(x_vals, timesteps, rewards))

        interpolated = np.array(interpolated)
        mean = interpolated.mean(axis=0)
        std = interpolated.std(axis=0)

        plt.plot(x_vals, mean, label=algo)
        plt.fill_between(x_vals, mean - std, mean + std, alpha=0.2)

    plt.title(f"{env} - Normalized Reward vs. Timesteps")
    plt.xlabel("Timesteps")
    plt.ylabel("Normalized Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{env.replace('-', '_')}_normalized_rewards.png")
    plt.close()



if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    for env in envs:
        plot_env_rewards(env)
