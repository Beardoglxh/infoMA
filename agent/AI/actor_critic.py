import torch
import torch.nn.functional as F
import numpy as np
from typing import List
import torch.nn as nn


def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class Actor(nn.Module):
    def __init__(self, args, use_mlp=False):
        super().__init__()
        encoding_size = 64
        policy_latent_size = 30
        action_space_n = 6
        input_size = args.input_size
        num_agents = args.n_agents
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device

        self.down_sampler = nn.Sequential(
            nn.Linear(input_size, encoding_size, device=device),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size, device=device),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(encoding_size, policy_latent_size, device=device),
            nn.ReLU(),
        )

        self.v_net = nn.Sequential(
            nn.Linear(encoding_size, 1, device=device),
            nn.Sigmoid(),
        )

        self.to_means = nn.Sequential(
            nn.Linear(policy_latent_size, action_space_n, device=device),
            nn.Tanh(),
        )

        if use_mlp:
            print('making a linear recurrent policy')
            self.recurr_policy = nn.Sequential(
                nn.Linear(policy_latent_size * num_agents, policy_latent_size, device=device),
                nn.Tanh(),
            )
            self.apply(init_params)
            self.init_recurr_policy()
        else:
            print('making a gru recurrent policy')
            self.apply(init_params)
            self.recurr_policy = nn.GRU(input_size=policy_latent_size, hidden_size=policy_latent_size, batch_first=True).to(device)

    def init_recurr_policy(self):
        self.recurr_policy[0].weight.data.copy_(torch.cat([torch.eye(30), torch.zeros(30, 30)], dim=1))

    def forward(self, observation, step, neighbors=None):
        if step == 0:
            return self.forward_initial(observation)
        elif step == 1:
            return self.forward_communicate(observation, neighbors)
        elif step == 2:
            return self.forward_probs(observation)
        else:
            raise Exception('Incorrect step number for forward prop, should be: 0,1,2')

    def forward_initial(self, observation):
        batch_size, current_dim = observation.shape
        input_tensor = torch.zeros((batch_size, current_dim)).to(observation.device)
        input_tensor[:, :current_dim] = observation

        encoded = self.down_sampler(input_tensor)
        policy_distribution = self.policy(encoded)
        # state_vals = self.v_net(encoded)
        state_vals = policy_distribution
        return (policy_distribution, state_vals)

    def forward_communicate(self, policy_dist, neighbors):
        """
        Modify latent vector distribution using neighboring distributions
        :param policy_dist: batchxlatent_size
        :param neighbors: batchxnum_neighborsxlatent_size
        :return: batchxlatent_size
        """
        # print(type(self.recurr_policy), isinstance(self.recurr_policy, nn.Linear))
        if isinstance(self.recurr_policy, nn.Sequential):
            batch, num_neighbors, latent_size = neighbors.shape
            flatten_neighbors = neighbors.reshape((batch, num_neighbors * latent_size))
            return self.recurr_policy(torch.cat([policy_dist, flatten_neighbors], dim=1))
        else:
            _, hn = self.recurr_policy(neighbors, policy_dist.unsqueeze(0))
            return hn.squeeze(0)

    def forward_probs(self, latent_vector):
        means = self.to_means(latent_vector)
        return means


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        obs_shape = 10
        agent_num = args.n_agents
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device
        self.max_action = args.high_action
        self.fc1 = nn.Linear(obs_shape * agent_num + args.action_shape, 64, device=device)
        self.fc2 = nn.Linear(64, 64, device=device)
        self.fc3 = nn.Linear(64, 64, device=device)
        self.q_out = nn.Linear(64, 1, device=device)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        action = action / self.max_action
        x = torch.cat([state, action], dim=1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
