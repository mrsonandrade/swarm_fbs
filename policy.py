# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn as nn

def calculate_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


class PositionalMapping(nn.Module):

    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):

        x = x * self.scale

        if self.L == 0:
            return x

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1) / self.scale


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mapping = PositionalMapping(input_dim=input_dim, L=0)

        h_dim = 128
        self.linear1 = nn.Linear(in_features=self.mapping.output_dim, out_features=h_dim, bias=True)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear4 = nn.Linear(in_features=h_dim, out_features=output_dim, bias=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.view([1, -1])
        x = self.mapping(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class ActorCritic(nn.Module):

    def __init__(self, input_dim, output_dim, device):
        super().__init__()

        self.device = device
        self.output_dim = output_dim

        self.actor = MLP(input_dim=input_dim, output_dim=output_dim)
        self.critic = MLP(input_dim=input_dim, output_dim=1)
        self.softmax = nn.Softmax(dim=-1)

        self.to(self.device)

        self.optimizer = optim.RMSprop(self.parameters(), lr=5e-5)

    def forward(self, x):
        y = self.actor(x)
        probs = self.softmax(y)
        value = self.critic(x)

        return probs, value

    def get_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs, value = self.forward(state)
        probs = probs[0, :]
        value = value[0]

        dist = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs).item()
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(
            torch.tensor(action, dtype=torch.long).to(self.device)
        )
        entropy = dist.entropy()

        return action, log_prob, value, entropy
