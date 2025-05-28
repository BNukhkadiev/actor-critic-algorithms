import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]
algos = ["REINFORCE", "MiniBatchREINFORCE", "PPO", "A2C", "TRPO", "ARS"]
results_dir = "results"
max_episodes = 500
window = 10

def load_rewards(env, algo):
    path = os.path.join(results_dir, algo, env)
    rewards = []
    if not os.path.exists(path):
        return []
    for fname in os.listdir(path):
        with open(os.path.join(path, fname), "r") as f:
            r = json.load(f)
            rewards.append(r[:max_episodes])
    return rewards

def smooth(x, w=10):
    return np.convolve(x, np.ones(w)/w, mode='valid')

os.makedirs("plots", exist_ok=True)

# --- 1. Combined reward plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, env in enumerate(envs):
    ax = axes.flat[idx]
    for algo in algos:
        runs = load_rewards(env, algo)
        if not runs:
            continue
        runs = [r[:max_episodes] for r in runs]
        min_len = min(len(r) for r in runs)
        runs = [r[:min_len] for r in runs]
        smoothed = np.array([smooth(r) for r in runs])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=algo)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_title(env)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)
fig.suptitle("Smoothed Reward Curves Across Environments", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/combined_rewards.png")
plt.close()

# --- 2. Final performance bar plot ---
summary = {}
for env in envs:
    summary[env] = {}
    for algo in algos:
        runs = load_rewards(env, algo)
        if runs:
            final_scores = [np.mean(r[-window:]) for r in runs if len(r) >= window]
            summary[env][algo] = np.mean(final_scores)
        else:
            summary[env][algo] = np.nan

bar_data = np.array([[summary[env].get(algo, np.nan) for algo in algos] for env in envs])
x = np.arange(len(envs))
width = 0.12

plt.figure(figsize=(14, 6))
for i, algo in enumerate(algos):
    plt.bar(x + i * width, bar_data[:, i], width=width, label=algo)

plt.xticks(x + width * len(algos) / 2, envs)
plt.ylabel("Final 10-Episode Avg Reward")
plt.title("Final Performance Comparison")
plt.legend(ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/final_performance_bar.png")
plt.close()

# --- 3. Ranking heatmap ---
rank_data = []
for env in envs:
    algo_scores = summary[env]
    scores = [(algo, algo_scores[algo]) for algo in algos if not np.isnan(algo_scores[algo])]
    scores.sort(key=lambda x: -x[1])  # descending
    ranking = {algo: rank+1 for rank, (algo, _) in enumerate(scores)}
    rank_data.append([ranking.get(algo, np.nan) for algo in algos])

rank_matrix = np.array(rank_data)

plt.figure(figsize=(10, 5))
sns.heatmap(rank_matrix, annot=True, fmt=".0f", cmap="coolwarm", xticklabels=algos, yticklabels=envs)
plt.title("Algorithm Ranking (1 = Best) per Environment")
plt.tight_layout()
plt.savefig("plots/rank_heatmap.png")
plt.close()
