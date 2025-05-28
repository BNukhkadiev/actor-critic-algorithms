from .reinforce import PolicyNetwork
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MiniBatchReinforce:
    def __init__(self, env_id="CartPole-v1", seed=0, lr=1e-2, batch_size=5):
        self.env = gym.make(env_id)
        self.env.reset(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.batch_size = batch_size

    def train(self, num_episodes=500):
        all_returns = []
        batch_log_probs, batch_returns = [], []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            log_probs, rewards = [], []

            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                probs = self.policy(obs_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                rewards.append(reward)

            returns = [sum(rewards[i:]) for i in range(len(rewards))]
            batch_log_probs.extend(log_probs)
            batch_returns.extend(returns)

            if (episode + 1) % self.batch_size == 0:
                loss = -torch.stack([lp * G for lp, G in zip(batch_log_probs, batch_returns)]).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_log_probs, batch_returns = [], []

            all_returns.append(sum(rewards))
        return all_returns
