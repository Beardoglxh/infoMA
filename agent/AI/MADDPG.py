import torch
import os
from agent.AI.actor_critic import Actor, Critic
# import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from scipy import stats

class MADDPG():
    def __init__(self, args, agent_id, agent_num, device):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.agent_num = agent_num
        self.plane_type = 0
        self.train_step = 0
        self.device = device
        # self.batch_size = args.batch_size

        # create the network
        self.actor_network = Actor(args).to(device)
        self.critic_network = Critic(args).to(device)

        # build up the target network
        self.actor_target_network = Actor(args)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

       # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, fitting_networks):
        train_data = {'o': [],
                      'initial': [],
                      'rewards': [],
                      'action': [],
                      'o_next': [],
                      'dones': []}
        for key in transitions:
            for t in transitions[key]:
                train_data[key].append(t.clone().detach().to(self.device))
        agent_id = self.agent_num

        batched_neighbors = []
        neighbor_next = []
        o_list = []
        o_next_list = []
        for id in range(self.args.n_agents):
            if agent_id != id:
                batched_neighbors.append(train_data['initial'][id].unsqueeze(1))
                neighbor, _ = fitting_networks[id].forward(train_data['o_next'][id], 0)
                neighbor_next.append(neighbor.unsqueeze(1))
            o_list.append(train_data['o'][id])
            o_next_list.append(train_data['o_next'][id])
        batched_neighbors = torch.cat(batched_neighbors,dim=1).detach()
        neighbor_next = torch.cat(neighbor_next, dim = 1).detach()
        # critic_loss
        # neughbor_next = neughbor_next.unsqueeze(1).detach()
        next_agent_dist, state_val = self.actor_target_network.forward(train_data['o_next'][agent_id], 0, None)
        # current_agent_dist = policy_initial[agent_ix]  # batch_seg x latent_shape
        next_latent_vec = self.actor_target_network.forward(next_agent_dist, 1, neighbor_next)
        next_final_action = self.actor_target_network.forward(next_latent_vec, 2, None)

        q_value = self.critic_network(o_list, train_data['action'][agent_id])
        q_next = self.critic_target_network(o_next_list, next_final_action)


        final_mask = torch.tensor(list(map(lambda s: s == True, train_data['dones'][agent_id])), dtype=torch.bool)
        q_next[final_mask] = torch.zeros(1)

        target_q = (train_data['rewards'][agent_id] + self.args.gamma * q_next).detach()
        critic_loss = (target_q - q_value).pow(2).mean()
        # actor_loss
        initial_dist, state_val = self.actor_network.forward(train_data['o'][agent_id], 0, None)
        current_agent_dist = initial_dist  # batch_seg x latent_shape
        latent_vec = self.actor_network.forward(current_agent_dist, 1, batched_neighbors)
        final_action_dist = self.actor_network.forward(latent_vec, 2, None)  # batch_seg x 4 (action_space)
        actor_loss = -self.critic_network(o_list, final_action_dist).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        self.train_step += 1


    def save_model(self, train_step):
        num = str(train_step)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


    def save_fitting_model(self, id):
        num = str(self.train_step // 10000)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.fitting_network[id].state_dict(), model_path + '/' + num + '_' + str(id) + '_fitting_params.pkl')
        # torch.save(self.actor_network.state_dict(), model_path + '/' + 'final_actor_params.pkl')
