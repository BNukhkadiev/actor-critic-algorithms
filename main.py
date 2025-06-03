from utils.runner import run_experiment
from algorithms.reinforce import Reinforce
from algorithms.minibatch_reinforce import MiniBatchReinforce
from algorithms.reinforce_continuous import ReinforceContinuous
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO, ARS

discrete_envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
# discrete_envs = []

continuous_envs = ["Pendulum-v1"]
# continuous_envs = []

# seeds = [0, 1, 2]
seeds = [3, 4, 5, 6, 7, 8, 9, 10]


for env_id in discrete_envs + continuous_envs:
    if env_id in discrete_envs:
        algorithms = {
            "REINFORCE": Reinforce,
            "MiniBatchREINFORCE": MiniBatchReinforce,
            "PPO": PPO,
            "A2C": A2C,
            "TRPO": TRPO,
            "ARS": ARS,
        }
    else:  # continuous environment
        algorithms = {
            "REINFORCE": ReinforceContinuous,
            "PPO": PPO,
            "A2C": A2C,
            "TRPO": TRPO,
            "ARS": ARS,
        }

    for algo_name, algo_cls in algorithms.items():
        for seed in seeds:
            run_experiment(env_id, algo_cls, algo_name, seed)
