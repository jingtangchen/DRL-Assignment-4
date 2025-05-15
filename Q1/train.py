import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_limit, device):
        self.buffer = deque(maxlen=buffer_limit)
        self.dev = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for s, a, r, s_prime, done in mini_batch:
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float).to(self.dev)
        a_batch = torch.tensor(a_lst, dtype=torch.float).to(self.dev)
        r_batch = torch.tensor(r_lst, dtype=torch.float).to(self.dev)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float).to(self.dev)
        done_batch = torch.tensor(done_mask_lst, dtype=torch.float).to(self.dev)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, action_dim)
        self.fc_std = nn.Linear(64, action_dim)

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.max_action = 2
        self.min_action = -2
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = torch.clamp(self.fc_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self(state)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, critic_lr):
        super().__init__()
        self.fc_s = nn.Linear(state_dim, 32)
        self.fc_a = nn.Linear(action_dim, 32)
        self.fc1 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)

    def forward(self, s, a):
        h1 = F.leaky_relu(self.fc_s(s))
        h2 = F.leaky_relu(self.fc_a(a))
        x = torch.cat([h1, h2], dim=-1)
        x = F.leaky_relu(self.fc1(x))
        return self.fc_out(x)


class SAC_Agent:
    def __init__(self):
        self.state_dim = 3
        self.action_dim = 1
        self.lr_pi = 0.001
        self.lr_q = 0.001
        self.gamma = 0.98
        self.batch_size = 200
        self.buffer_limit = 100000
        self.tau = 0.005
        self.init_alpha = 0.01
        self.target_entropy = -self.action_dim
        self.lr_alpha = 0.005
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = ReplayBuffer(self.buffer_limit, self.DEVICE)

        self.log_alpha = torch.tensor(np.log(self.init_alpha), requires_grad=True, device=self.DEVICE)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

        self.PI = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi).to(self.DEVICE)
        self.Q1 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2 = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q1_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q2_target = QNetwork(self.state_dim, self.action_dim, self.lr_q).to(self.DEVICE)
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def choose_action(self, state):
        with torch.no_grad():
            action, _ = self.PI.sample(state.to(self.DEVICE))
        return action

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        with torch.no_grad():
            a_prime, log_prob = self.PI.sample(s_prime)
            entropy = -self.log_alpha.exp() * log_prob
            q_target = torch.min(self.Q1_target(s_prime, a_prime), self.Q2_target(s_prime, a_prime))
            return r + self.gamma * done * (q_target + entropy)

    def train_agent(self):
        s, a, r, s_prime, done = self.memory.sample(self.batch_size)
        td_target = self.calc_target((s, a, r, s_prime, done))

        q1_loss = F.smooth_l1_loss(self.Q1(s, a), td_target)
        self.Q1.optimizer.zero_grad()
        q1_loss.mean().backward()
        self.Q1.optimizer.step()

        q2_loss = F.smooth_l1_loss(self.Q2(s, a), td_target)
        self.Q2.optimizer.zero_grad()
        q2_loss.mean().backward()
        self.Q2.optimizer.step()

        a_pi, log_prob = self.PI.sample(s)
        entropy = -self.log_alpha.exp() * log_prob
        q = torch.min(self.Q1(s, a_pi), self.Q2(s, a_pi))
        pi_loss = -(q + entropy)
        self.PI.optimizer.zero_grad()
        pi_loss.mean().backward()
        self.PI.optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        for param, target_param in zip(self.Q1.parameters(), self.Q1_target.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
        for param, target_param in zip(self.Q2.parameters(), self.Q2_target.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    agent = SAC_Agent()

    for ep in range(200):
        state, _ = env.reset()
        done, score = False, 0.0

        while not done:
            action = agent.choose_action(torch.FloatTensor(state))
            action = action.cpu().numpy()
            next_state, reward, truncated, terminated, _ = env.step(action)
            done = terminated or truncated
            agent.memory.put((state, action, reward, next_state, done))
            state = next_state
            score += reward

            if agent.memory.size() > 1000:
                agent.train_agent()

        print(f"EP:{ep}, Score:{score:.1f}")

    torch.save(agent.PI.state_dict(), "sac_actor_final.pt")
