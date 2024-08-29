import numpy as np
import torch
from env.env_cmd import CmdEnv as env_cmd
from agent.AI.MADDPG import MADDPG


class Planes_Policy:
    def __init__(self, args, device, k_levels=1):
        self.num_agents = args.n_agents
        self.device = device
        self.args = args
        self.agents = [MADDPG(self.args, agent_ix, i, device) for i, agent_ix in enumerate(args.plane_id)]
        self.k_levels = k_levels
        self.noise_rate = args.noise_rate
        self.epsione = args.epsione

    def to(self, device):
        for agent in self.agents:
            agent.actor_network.to(device)
            agent.critic_network.to(device)
            agent.actor_target_network.to(device)
            agent.critic_target_network.to(device)

    def save_policy(self, time):
        for agent in self.agents:
            agent.save_model(time)

    def input_arrange(self, plane_info, enemy_plane):
        obervations = []
        enemy_info = torch.tensor([[]]).to(self.device)

        for plane in enemy_plane:
            obervation = []
            if plane["Availability"] == 1:
                obervation.append(plane["Heading"])
                obervation.append(plane["Speed"] / 1e+2)
                obervation.append(plane["X"] / 1e+3)
                obervation.append(plane["Y"] / 1e+3)
                obervation.append(plane["Alt"] / 1e+3)
                # obervation.append(plane.Speed)
                # obervation.append(plane.X)
                # obervation.append(plane.Y)
                # obervation.append(plane.Z)
                obervation = torch.tensor(obervation).unsqueeze(dim=0).to(self.device)
            enemy_info = torch.cat((enemy_info, obervation), dim=1)

        # if len(enemy_plane) < 3:
        #     print("HEllo")
        #     tmp = torch.zeros((1, 5 * (3 - len(enemy_plane)))).to(self.device)
        #     enemy_info = torch.cat((enemy_info, tmp), dim=1)

        for plane in plane_info:
            obervation = []
            if plane.Availability == 1:
                obervation.append(plane.Heading)
                obervation.append(plane.Speed / 1e+2)
                obervation.append(plane.X / 1e+3)
                obervation.append(plane.Y / 1e+3)
                obervation.append(plane.Z / 1e+3)
                # obervation.append(plane.Speed)
                # obervation.append(plane.X)
                # obervation.append(plane.Y)
                # obervation.append(plane.Z)
                obervation = torch.tensor(obervation).unsqueeze(dim=0).to(self.device)
                obervation = torch.cat((obervation, enemy_info), dim=1)
            obervations.append(obervation)

        return obervations

    def __call__(self, plane_info, enemy_plane, global_obs, noise_rate=0, epsilon=0):
        noise_rate = self.noise_rate
        epsilon = self.epsione
        observations = self.input_arrange(plane_info, enemy_plane)
        k_levels = self.k_levels
        # num_batches = observations[0].shape[0]
        cmd_list = []
        policy_initial = []
        state_vals = []
        for agent_ix in range(self.num_agents):
            initial_dist, state_val = self.agents[agent_ix].actor_network.forward(observations[agent_ix], 0, None)
            state_vals.append(state_val)
            policy_initial.append(initial_dist)

        for k in range(0, k_levels):
            output_dist = []
            # output_dist = torch.zeros(size=policy_initial.shape)
            for agent_ix in range(self.num_agents):
                batched_neighbors = []
                for id in range(self.num_agents):
                    if id == agent_ix:
                        continue
                    else:
                        batched_neighbors.append(policy_initial[id].unsqueeze(0))
                batched_neighbors = torch.cat(batched_neighbors, dim=1)
                current_agent_dist = policy_initial[agent_ix]  # batch_seg x latent_shape
                latent_vec = self.agents[agent_ix].actor_network.forward(current_agent_dist, 1, batched_neighbors)
                output_dist.append(latent_vec)  # batch_seq x latent_shape
                # output_dist[:, agent_ix] = latent_vec
            policy_initial = output_dist

        final_actions = []
        for agent_ix in range(0, self.num_agents):
            final_agent_latent = policy_initial[agent_ix]
            final_action_dist = self.agents[agent_ix].actor_network.forward(final_agent_latent, 2,
                                                                            None)  # batch_seg x 4 (action_space)
            final_actions.append(final_action_dist)  # batch_seg x 4 (action_space)

        results = []
        for agent_ix in range(0, self.num_agents):
            action = final_actions[agent_ix]
            # for action_id, action in enumerate(final_actions[agent_ix]):
            if np.random.uniform() < epsilon:
                result = np.random.uniform(-1, 1, self.args.action_shape).astype(np.float32)
                # result = torch.from_numpy(result).float()
            else:
                result = action.squeeze(0).cpu().detach().numpy()
            # result = action.cpu().detach().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(self.args.action_shape).astype(
                np.float32)  # gaussian noise
            result += noise
            result = np.clip(result, -1, 1)
            # results[agent_ix].append(torch.from_numpy(result).float().unsqueeze(0))
            # results[agent_ix] = torch.cat(results[agent_ix], dim=0)
            results.append(result)

        for i, agent in enumerate(self.agents):
            action = results[i]
            if agent.plane_type == 1:
                x = action[0] * 149999
                y = action[1] * 149999
                z = (1 + action[2]) * (self.args.plane_ability["leader"]["height"][1] -
                                       self.args.plane_ability["leader"]["height"][0]) / 2 + \
                    self.args.plane_ability["leader"]["height"][0]
                speed = (1 + action[3]) * (
                            self.args.plane_ability["leader"]["speed"][1] - self.args.plane_ability["leader"]["speed"][
                        0]) / 2 + self.args.plane_ability["leader"]["speed"][0]
                accelerate = (1 + action[4]) * (self.args.plane_ability["leader"]["accelerate"]) / 2
                overload = (1 + action[5]) * self.args.plane_ability["leader"]["overload"] / 2
            else:
                x = action[0] * 149999
                y = action[1] * 149999
                z = (1 + action[2]) * (self.args.plane_ability["uav"]["height"][1] -
                                       self.args.plane_ability["uav"]["height"][0]) / 2 + \
                    self.args.plane_ability["uav"]["height"][0]
                speed = (1 + action[3]) * (
                        self.args.plane_ability["uav"]["speed"][1] - self.args.plane_ability["uav"]["speed"][
                    0]) / 2 + self.args.plane_ability["uav"]["speed"][0]
                accelerate = (1 + action[4]) * (self.args.plane_ability["uav"]["accelerate"]) / 2
                overload = (1 + action[5]) * self.args.plane_ability["uav"]["overload"] / 2
            cmd_list.append(
                env_cmd.make_linepatrolparam(agent.agent_id, [{"X": x, "Y": y, "Z": z}], speed, accelerate, overload))
        self.noise_rate = max(0.05, self.noise_rate - 0.0000005)
        self.epsione = max(0.05, self.epsione - 0.0000005)
        return cmd_list, observations, state_vals, results

    def trainer(self, transitions):
        fitting_networks = []
        for plane in self.agents:
            fitting_networks.append(plane.actor_target_network)
        for plane in self.agents:
            plane.train(transitions, fitting_networks)
