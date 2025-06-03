import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)
    
    
class Reinforce:
    def __init__(self, env_id="CartPole-v1", seed=0, lr=1e-2):
        self.env = gym.make(env_id)
        self.env.reset(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def train(self, num_episodes=500):
        timestep = 0
        timestep_rewards = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            log_probs, rewards = [], []

            done = False
            ep_len = 0
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                probs = self.policy(obs_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                rewards.append(reward)
                ep_len += 1

            # Policy update
            returns = [sum(rewards[i:]) for i in range(len(rewards))]
            loss = -torch.stack([lp * G for lp, G in zip(log_probs, returns)]).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            timestep += ep_len
            ep_return = sum(rewards)
            timestep_rewards.append([timestep, ep_return])

        return timestep_rewards
