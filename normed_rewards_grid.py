import os
import matplotlib.pyplot as plt
from PIL import Image

# Directory containing normalized reward plots
PLOT_DIR = "plots"
# List of environments matching the file naming convention
ENVIRONMENTS = [
    "CartPole_v1",
    "Acrobot_v1",
    "MountainCar_v0",
    "Pendulum_v1"
]

# Grid layout (adjust columns if needed)
n_cols = 2
n_rows = (len(ENVIRONMENTS) + n_cols - 1) // n_cols

# Create subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
axes = axes.flatten()

for i, env in enumerate(ENVIRONMENTS):
    file_path = os.path.join(PLOT_DIR, f"{env}_normalized_rewards.png")
    if os.path.exists(file_path):
        img = Image.open(file_path)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(env.replace("_", "-"), fontsize=10)
    else:
        axes[i].axis("off")
        axes[i].set_title(f"{env}\n[missing]", fontsize=9, color='red')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.suptitle("Normalized Rewards Across Environments", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
fig.savefig("plots/normalized_rewards_grid_summary.png", dpi=300)
