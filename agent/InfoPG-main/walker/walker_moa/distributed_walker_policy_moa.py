import torch.nn as nn
import torch
import os

class Combined_MOAPong_Helper:
    def __init__(self, device):
        self.device = device
        self.num_agents = 2
        self.agents = [Walker_MOAPolicyHelper().to(device) for _ in range(self.num_agents)]

    def parameters(self):
        return [model.parameters() for model in self.agents]

    def state_dicts(self):
        return [model.state_dict() for model in self.agents]

    def load_state_dicts(self, state_dicts):
        return [model.load_state_dict(state_dicts[ix]) for ix, model in enumerate(self.agents)]

    def __call__(self, observations, prev_neighbor_action_dists):
        """

        :param adj_matrix: torch.Tensor of shape: batch_seg x agents x agents
        :param observations: list of length agents, where element is: batch_seg x obs_shape
        :return:
        """
        output_actions = []
        output_values = []
        output_distribution_neighbors = []
        for agent_ix in range(0, self.num_agents):
            action_dist, v, action_other_paddle = self.agents[agent_ix](observations[agent_ix], prev_neighbor_action_dists[:,agent_ix].unsqueeze(1))
            output_actions.append(action_dist)
            output_values.append(v)
            output_distribution_neighbors.append(action_other_paddle)
        return output_actions, output_values, output_distribution_neighbors

    def to(self, device):
        pass

    def eval_experiment(self, experiment_name):
        with open(os.path.join('experiments', 'walker_moa', experiment_name, 'combined_model.pt'), 'rb') as f:
            d = torch.load(f, map_location=self.device)
        self.load_state_dicts(d['policy'])


class Walker_MOAPolicyHelper(nn.Module):
    def __init__(self):
        super().__init__()
        encoding_size = 256
        policy_latent_size = 30
        action_space_n = 4

        self.down_sampler = nn.Sequential(
            nn.Linear(31, encoding_size),
            nn.ReLU(),
        )

        self.fc_e = nn.Sequential(
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU(),
        )

        self.pi_e = nn.Sequential(
            nn.Linear(encoding_size, policy_latent_size),
            nn.ReLU(),
            nn.Linear(policy_latent_size, action_space_n),
            nn.Tanh(),
        )

        self.v_e = nn.Sequential(
            nn.Linear(encoding_size, 1),
            nn.Tanh(),
        )

        self.fc_moa = nn.Sequential(
            nn.Linear(encoding_size, policy_latent_size),
            nn.ReLU(),
            nn.Linear(policy_latent_size, action_space_n),
            nn.ReLU(),
        )

        self.gru = nn.GRU(action_space_n, action_space_n, batch_first=True)

    def forward(self, input_state, prev_neighbor_action_dists):
        down_samp = self.down_sampler(input_state)

        latent_e = self.fc_e(down_samp)
        output_dist = self.pi_e(latent_e)
        v = self.v_e(latent_e)

        latent_moa = self.fc_moa(down_samp)
        (_, h_n) = self.gru(prev_neighbor_action_dists, latent_moa.unsqueeze(0))
        return output_dist, v, h_n.squeeze(0)