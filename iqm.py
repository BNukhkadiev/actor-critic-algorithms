import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

# Configuration
envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]

algos = ["REINFORCE", "MiniBatchREINFORCE", "PPO", "A2C", "TRPO", "ARS"]
results_dir = "results"
frame_bins = np.linspace(0, 50000, 300)

def load_normalized_interpolated(env):
    raw_data = {}
    all_rewards = []

    for algo in algos:
        algo_path = os.path.join(results_dir, algo, env)
        if not os.path.exists(algo_path):
            continue

        runs = []
        for fname in os.listdir(algo_path):
            with open(os.path.join(algo_path, fname), "r") as f:
                r = json.load(f)
                if isinstance(r[0], list) and len(r[0]) == 2:
                    ts, rs = zip(*r)
                    runs.append((np.array(ts), np.array(rs)))
                    all_rewards.extend(rs)

        if runs:
            raw_data[algo] = runs

    # Normalize
    min_r, max_r = min(all_rewards), max(all_rewards)
    algo_to_interpolated = {}

    for algo, runs in raw_data.items():
        interpolated = []
        for ts, rs in runs:
            norm_rs = (rs - min_r) / (max_r - min_r + 1e-8)
            interp = np.interp(frame_bins, ts, norm_rs)
            interpolated.append(interp)
        algo_to_interpolated[algo] = np.array(interpolated)  # shape: (num_seeds, num_timesteps)

    return algo_to_interpolated

def iqm(data, axis=0):
    """Interquartile Mean along axis=0 (per timestep)"""
    q25 = np.quantile(data, 0.25, axis=axis, keepdims=True)
    q75 = np.quantile(data, 0.75, axis=axis, keepdims=True)
    mask = (data >= q25) & (data <= q75)
    masked = np.where(mask, data, np.nan)
    return np.nanmean(masked, axis=axis)

def bootstrap_iqm_ci(data, n_resamples=500, ci=0.95):
    """Bootstrap confidence interval for IQM (axis=0 = across runs)"""
    result = bootstrap(
        (data,),
        statistic=iqm,
        axis=0,
        confidence_level=ci,
        n_resamples=n_resamples,
        method="basic",
        vectorized=False  # Important!
    )
    return result.confidence_interval.low, result.confidence_interval.high

def plot_iqm_curve(env):
    data = load_normalized_interpolated(env)
    plt.figure(figsize=(10, 6))

    for algo, scores in data.items():  # scores shape: (num_runs, num_timesteps)
        iqm_curve = iqm(scores, axis=0)
        ci_low, ci_high = bootstrap_iqm_ci(scores)

        plt.plot(frame_bins, iqm_curve, label=algo)
        plt.fill_between(frame_bins, ci_low, ci_high, alpha=0.2)

    plt.title(f"{env} - IQM Sample Efficiency (95% CI)")
    plt.xlabel("Timesteps")
    plt.ylabel("Normalized Reward (IQM)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{env.replace('-', '_')}_iqm_sample_efficiency.png")
    plt.close()

if __name__ == "__main__":
    for env in envs:
        plot_iqm_curve(env)
