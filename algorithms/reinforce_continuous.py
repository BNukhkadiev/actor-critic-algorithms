import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.mu_head = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learnable σ

    def forward(self, x):
        features = self.net(x)
        mu = self.mu_head(features)
        std = torch.exp(self.log_std)
        return mu, std

    def sample_action(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1)

class ReinforceContinuous:
    def __init__(self, env_id, seed=0, lr=1e-3, gamma=0.99):
        self.env = gym.make(env_id)
        self.env.reset(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        self.policy = GaussianPolicy(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def train(self, num_episodes=500):
        all_rewards = []

        for episode in range(num_episodes):
            log_probs = []
            rewards = []

            state, _ = self.env.reset()
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action, log_prob = self.policy.sample_action(state_tensor)
                clipped_action = torch.clamp(action, float(self.env.action_space.low[0]), float(self.env.action_space.high[0]))
                next_state, reward, terminated, truncated, _ = self.env.step(clipped_action.detach().numpy())

                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state
                done = terminated or truncated

            all_rewards.append(sum(rewards))
            returns = self._compute_returns(rewards)

            loss = -torch.stack([
                log_prob * ret for log_prob, ret in zip(log_probs, returns)
            ]).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return all_rewards

    def _compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
