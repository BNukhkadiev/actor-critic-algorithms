import os
import matplotlib.pyplot as plt
from PIL import Image

# Directory where your IQM plots are stored
PLOT_DIR = "plots"
# File pattern: "{env_id}_iqm_sample_efficiency.png"
ENVIRONMENTS = [
    "CartPole_v1",
    "Acrobot_v1",
    "MountainCar_v0",
    "Pendulum_v1"
]

# Grid layout (adjust if you have more/fewer envs)
n_cols = 2
n_rows = (len(ENVIRONMENTS) + n_cols - 1) // n_cols

# Plot settings
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
axes = axes.flatten()

for i, env_id in enumerate(ENVIRONMENTS):
    file_path = os.path.join(PLOT_DIR, f"{env_id}_iqm_sample_efficiency.png")
    if os.path.exists(file_path):
        img = Image.open(file_path)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(env_id.replace("_", "-"), fontsize=10)
    else:
        axes[i].axis("off")
        axes[i].set_title(f"{env_id}\n[missing]", fontsize=9, color='red')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.suptitle("IQM Sample Efficiency Across Environments", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
fig.savefig("plots/iqm_grid_summary.png", dpi=300)


