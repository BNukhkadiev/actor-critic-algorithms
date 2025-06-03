import os
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import bootstrap

results_dir = "results"
algos = ["REINFORCE", "MiniBatchREINFORCE", "PPO", "A2C", "TRPO", "ARS"]
envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]

frame_bins = np.linspace(0, 50000, 300)

def load_data(env):
    algo_to_runs = {}
    all_rewards = []

    for algo in algos:
        path = os.path.join(results_dir, algo, env)
        if not os.path.exists(path):
            continue

        runs = []
        for fname in os.listdir(path):
            with open(os.path.join(path, fname), "r") as f:
                data = json.load(f)
                if isinstance(data[0], list) and len(data[0]) == 2:
                    timesteps, rewards = zip(*data)
                    runs.append((np.array(timesteps), np.array(rewards)))
                    all_rewards.extend(rewards)
        if runs:
            algo_to_runs[algo] = runs

    return algo_to_runs, all_rewards

def bootstrap_ci(data, confidence_level=0.95):
    data = np.array(data)
    res = bootstrap((data,), np.mean, confidence_level=confidence_level, n_resamples=500, method='basic', axis=0)
    return res.confidence_interval.low, res.confidence_interval.high

def plot_sample_efficiency_with_ci(env):
    algo_to_runs, all_rewards = load_data(env)
    if not all_rewards:
        print(f"[WARNING] No data for {env}")
        return

    min_r, max_r = min(all_rewards), max(all_rewards)
    plt.figure(figsize=(10, 6))

    for algo, runs in algo_to_runs.items():
        interpolated = []
        for ts, rs in runs:
            norm_rs = (rs - min_r) / (max_r - min_r + 1e-8)
            interp = np.interp(frame_bins, ts, norm_rs)
            interpolated.append(interp)
        interpolated = np.array(interpolated)

        # Compute point estimate and bootstrap CI
        mean = interpolated.mean(axis=0)
        ci_low, ci_high = bootstrap_ci(interpolated)

        plt.plot(frame_bins, mean, label=algo)
        plt.fill_between(frame_bins, ci_low, ci_high, alpha=0.2)

    plt.title(f"{env} - Sample Efficiency (95% CI)")
    plt.xlabel("Timesteps")
    plt.ylabel("Normalized Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{env.replace('-', '_')}_sample_efficiency_ci.png")
    plt.close()

if __name__ == "__main__":
    for env in envs:
        plot_sample_efficiency_with_ci(env)
