import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]
algos = ["REINFORCE", "MiniBatchREINFORCE", "PPO", "A2C", "TRPO", "ARS"]
results_dir = "results"
max_episodes = 500
final_fraction = 0.1  # Use last 10% of timesteps

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

def normalize_and_interpolate(runs, x_vals, min_r, max_r):
    normalized = []
    for run in runs:
        run = [(t, r) for t, r in run if isinstance(r, (int, float))]  # ensure valid
        if not run:
            continue
        timesteps, rewards = zip(*run)
        rewards = [(r - min_r) / (max_r - min_r + 1e-8) for r in rewards]
        interp = np.interp(x_vals, timesteps, rewards)
        normalized.append(interp)
    return np.array(normalized)

# Step 1: Normalize all data and collect final scores
summary = {}
x_vals = np.linspace(0, 50000, 300)
final_n = int(len(x_vals) * final_fraction)

for env in envs:
    all_rewards = []

    # Collect raw rewards to compute normalization bounds
    raw = {}
    for algo in algos:
        runs = load_rewards(env, algo)
        if not runs:
            continue
        raw[algo] = runs
        all_rewards.extend([r for run in runs for _, r in run])

    if not all_rewards:
        continue

    min_r = min(all_rewards)
    max_r = max(all_rewards)

    summary[env] = {}
    for algo in algos:
        if algo not in raw:
            summary[env][algo] = np.nan
            continue
        normed = normalize_and_interpolate(raw[algo], x_vals, min_r, max_r)
        if len(normed) == 0:
            summary[env][algo] = np.nan
            continue
        final_scores = normed[:, -final_n:].mean(axis=1)
        summary[env][algo] = np.mean(final_scores)

# Step 2: Plot bar chart
os.makedirs("plots", exist_ok=True)
bar_data = np.array([[summary[env].get(algo, np.nan) for algo in algos] for env in envs])
x = np.arange(len(envs))
width = 0.12

plt.figure(figsize=(14, 6))
for i, algo in enumerate(algos):
    plt.bar(x + i * width, bar_data[:, i], width=width, label=algo)

plt.xticks(x + width * len(algos) / 2, envs)
plt.ylabel("Final Avg Normalized Reward (Last 10%)")
plt.title("Final Normalized Performance Comparison")
plt.legend(ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/final_normalized_bar.png")
plt.close()

# Step 3: Ranking heatmap
rank_data = []
for env in envs:
    algo_scores = summary[env]
    scores = [(algo, algo_scores[algo]) for algo in algos if not np.isnan(algo_scores[algo])]
    scores.sort(key=lambda x: -x[1])
    ranking = {algo: rank + 1 for rank, (algo, _) in enumerate(scores)}
    rank_data.append([ranking.get(algo, np.nan) for algo in algos])

rank_matrix = np.array(rank_data)

plt.figure(figsize=(10, 5))
sns.heatmap(rank_matrix, annot=True, fmt=".0f", cmap="coolwarm", xticklabels=algos, yticklabels=envs)
plt.title("Algorithm Ranking (1 = Best) per Environment")
plt.tight_layout()
plt.savefig("plots/final_normalized_ranking.png")
plt.close()


# Step 4: Combined normalized reward plots per environment
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, env in enumerate(envs):
    ax = axes.flat[idx]
    all_rewards = []
    raw = {}

    for algo in algos:
        runs = load_rewards(env, algo)
        if not runs:
            continue
        raw[algo] = runs
        all_rewards.extend([r for run in runs for _, r in run])

    if not all_rewards:
        ax.set_title(env + " (no data)")
        continue

    min_r = min(all_rewards)
    max_r = max(all_rewards)

    for algo in algos:
        if algo not in raw:
            continue
        normed = normalize_and_interpolate(raw[algo], x_vals, min_r, max_r)
        if len(normed) == 0:
            continue
        mean = normed.mean(axis=0)
        std = normed.std(axis=0)
        ax.plot(x_vals, mean, label=algo)
        ax.fill_between(x_vals, mean - std, mean + std, alpha=0.2)

    ax.set_title(env)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Normalized Reward")
    ax.grid(True)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)
fig.suptitle("Normalized Reward Curves Across Environments", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/combined_normalized_rewards.png")
plt.close()
