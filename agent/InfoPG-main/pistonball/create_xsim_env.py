import datetime
import os

from get_obs import startXsim
from xsim_config import address
from policy_base import Experience
import numpy as np
import torch
from torch.distributions import Categorical
from plane_decision import Decision
from reward_1229 import Reward
from time import sleep
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3


class planeinfo:
    def __init__(self, agent):
        # 平台编号
        self.ID = agent['ID']
        # x轴坐标(浮点型, 单位: 米, 下同)
        self.X = agent['X']
        # y轴坐标
        self.Y = agent['Y']
        # z轴坐标
        self.Z = agent['Alt']
        # 航向(浮点型, 单位: 度, [0-360])
        self.Pitch = agent['Pitch']
        # 横滚角
        self.Roll = agent['Roll']
        # 航向, 即偏航角
        self.Heading = agent['Heading']
        # 速度
        self.Speed = agent['Speed']
        # 当前可用性
        self.Availability = agent['Availability']
        # 类型
        self.Type = agent['Type']
        # 仿真时间
        self.CurTime = agent['CurTime']
        # 军别信息
        self.Identification = agent['Identification']
        # 是否被锁定
        self.IsLocked = agent['IsLocked']
        # 剩余弹药
        self.LeftWeapon = agent['LeftWeapon']

        # 坐标
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}
        # 被几发导弹瞄准
        self.num_locked_missile = 0

    def can_attack(self, dis):
        performance = self.performance()
        if performance["launch_range"] > dis:
            return True
        return False

    def performance(self):
        para = {}
        if self.Type == 1:
            para["move_max_speed"] = 400
            para["move_max_acc"] = 1
            para["move_max_g"] = 6
            para["area_max_alt"] = 14000
            para["attack_range"] = 0.8
            para["launch_range"] = 80000 * 0.8
        else:
            para["move_max_speed"] = 300
            para["move_max_acc"] = 2
            para["move_max_g"] = 6
            para["area_max_alt"] = 10000
            para["attack_range"] = 1
            para["launch_range"] = 60000
        return para


class CreateXsim:
    def __init__(self, batch_size: int, env_params, seed=None):
        print("Try to create XsimEnv")
        self.BATCH_SIZE = batch_size
        self.MAX_CYCLES = env_params["max_cycles"]
        self.address = address["ip"] + ":" + str(address["port"])
        self.envs = [startXsim(self.address) for i in range(0, self.BATCH_SIZE)]
        self.obs = self.envs[0].getObs()
        self.filt_my_plane = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 2 or x["ID"] == 11 or x["ID"] == 12 or x["ID"] == 23]
        self.bait_my_plane = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 20 or x["ID"] == 21 or x["ID"] == 22]
        self.enemy_plane = [x for x in self.obs["blue"]["platforminfos"] if x["ID"] == 6 or x["ID"] == 19 or x["ID"] == 26]
        self.encoding = self.envs[0].obs_encoding(self.filt_my_plane, self.enemy_plane)
        self.N_AGENTS = len(self.filt_my_plane)
        self.AGENT_NAMES = np.array([x["Name"] for x in self.filt_my_plane])
        self.adj_matrix = np.zeros((self.BATCH_SIZE, len(self.AGENT_NAMES), len(self.AGENT_NAMES)))

        for i in range(0, self.BATCH_SIZE):
            for j in range(0, len(self.AGENT_NAMES)):
                if j - 1 >= 0:
                    self.adj_matrix[i][j][j - 1] = 1
                if j + 1 < len(self.AGENT_NAMES):
                    self.adj_matrix[i][j][j + 1] = 1
        self.update_adj_matrix()
        self.ACTION_SPACE = 5  # 待确认
        self.REWARD_SCALE = 1  # 待确认
        self.OBS_SHAPE = (1, 20)
        self.decision = Decision()
        self.reward = Reward(self.class_my_plane, self.class_enemy_plane)
        self.experiment_name = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        self.path = os.path.join("model", self.experiment_name)
        os.makedirs(self.path)

    def loop(self, user_params, policies, optimizers, schedulers):
        device = user_params['device']
        # alex_encoder = Encoder(device)
        epochs = user_params['epochs']
        verbose = user_params['verbose']
        communicate = user_params['communicate']
        max_grad_norm = user_params['max_grad_norm']
        time_penalty = user_params['time_penalty']
        early_reward_benefit = user_params['early_reward_benefit']
        if 'k-levels' in user_params.keys():
            k_levels = user_params['k-levels']
        else:
            k_levels = 1
        print('**Using K-Levels: ', k_levels)

        if 'consensus_update' in user_params.keys():
            consensus_update = user_params['consensus_update']
        else:
            consensus_update = False
        print('**Using Consensus-Update: ', consensus_update)

        summary_stats = []
        for epoch in range(0, epochs):
            if verbose:
                print('Epoch: %s' % (epoch))
            observations = self.batch_reset()  # TODO 得到观测值
            for agent in observations:
                tmp = torch.tensor(observations[agent]).reshape(self.OBS_SHAPE).to(
                    dtype=torch.float32)
                observations[agent] = tmp.clone().detach()
            # observations = self.reshape_obs(observations)
            for step in range(self.MAX_CYCLES):
                num_left_batches = np.count_nonzero(
                    self.DONE_ENVS == False)  # TODO 每一个batch是否结束就是看这个DONE_ENVS   number of environments that aren't done
                left_batches = np.where(self.DONE_ENVS == False)[
                    0]  # indeces of batches that aren't done (length is num_left_batches)
                if num_left_batches == 0:
                    break

                memory = self.initialize_memory()

                # with torch.no_grad():
                #     for agent in self.AGENT_NAMES:
                #         observations[agent] = alex_encoder(observations[agent])  # TODO 输入到Alexnet里展平到了(4, 4096)

                policy_initial = {}
                for agent in self.AGENT_NAMES:
                    initial_policy_distribution, state_val = policies[agent].forward(observations[agent], 0,
                                                                                     None)  # TODO (4, 4096)->(4, 20)
                    for batch_ix in range(0, num_left_batches):
                        memory[agent][left_batches[batch_ix]].state_val = state_val[batch_ix]  # TODO 这里把state_value存起来了
                    policy_initial[agent] = initial_policy_distribution

                actions = {agent_name: [-1 for _ in range(0, num_left_batches)] for agent_name in self.AGENT_NAMES}
                if communicate:  # TODO 初始策略进行通讯
                    policy_initial = self.k_level_communication(policies, policy_initial, num_left_batches,
                                                                left_batches, k_levels)

                for agent in self.AGENT_NAMES:
                    final_policy_distribution = policies[agent].forward(policy_initial[agent], 2, None).to(
                        'cpu').clone()  # TODO (4, 20)->(4, 3) softmax
                    distribution = Categorical(probs=final_policy_distribution)
                    batch_action = distribution.sample()
                    batched_log_prob = distribution.log_prob(batch_action)  # TODO 根据上一步采样的结果取相对应的对数值
                    for batch_ix in range(0, num_left_batches):
                        action = batch_action[batch_ix].item()  # 当前智能体每个batch的动作
                        log_prob = batched_log_prob[batch_ix]
                        actions[agent][batch_ix] = action
                        actual_batch_number = left_batches[batch_ix]
                        memory[agent][actual_batch_number].action = action  # 对应的数据存起来
                        memory[agent][actual_batch_number].log_prob = log_prob
                        memory[agent][actual_batch_number].policy_distribution = final_policy_distribution[batch_ix]

                next_observations, rewards, dones = self.batch_step(observations, actions, step, time_penalty,
                                                                    early_reward_benefit)
                self.add_rewards_to_memory(policies, memory, rewards, num_left_batches, left_batches)
                observations = self.conclude_step(next_observations, dones)

            for agent in self.AGENT_NAMES:
                optimizers[agent].zero_grad(set_to_none=False)
                policies[agent].set_batched_storage(self.BATCH_SIZE)

            epoch_data, team_iterations = self.compute_epoch_data(policies, verbose=True)
            summary_stats.append(epoch_data)
            if verbose:
                print('\t *Team Mean Iterations: %s' % (team_iterations))

            for agent in self.AGENT_NAMES:
                torch.nn.utils.clip_grad_norm_(policies[agent].parameters(), max_grad_norm)

            self.conclude_epoch(policies, optimizers, schedulers)
            if consensus_update:
                self.consensus_update(policies)
            self.save_model(policies, optimizers, epoch)

        return policies, optimizers, summary_stats

    def save_model(self, policies, optimizers, epoch):
        for agent in policies.keys():
            torch.save({
                'policy': policies[agent].state_dict(),
                'optimizer': optimizers[agent].state_dict()
            }, os.path.join(self.path, str(epoch) + '_' + agent + '.pt'))

    def reshape_obs(self, obs):
        return torch.reshape(obs, self.OBS_SHAPE)

    def update_adj_matrix(self):
        pass

    def get_agent_names(self):
        return self.AGENT_NAMES

    def batch_reset(self):
        self.DONE_ENVS = np.array([False for _ in range(0, self.BATCH_SIZE)])
        self.reward.reset()
        ret_obs = {agent_name: np.zeros((self.BATCH_SIZE,) + self.OBS_SHAPE) for agent_name in
                   self.AGENT_NAMES}
        for batch_ix in range(0, self.BATCH_SIZE):
            self.obs = self.envs[batch_ix].system_reset()
            while True:
                if self.obs is None:
                    sleep(0.2)
                    self.obs = self.envs[0].getObs()
                    print("can not get obs now")
                    continue
                else:
                    break
            self.filt_my_plane = [x for x in self.obs["red"]["platforminfos"] if
                                  x["ID"] == 2 or x["ID"] == 11 or x["ID"] == 12 or x["ID"] == 23]
            self.bait_my_plane = [x for x in self.obs["red"]["platforminfos"] if
                                  x["ID"] == 20 or x["ID"] == 21 or x["ID"] == 22]
            self.enemy_plane = [x for x in self.obs["blue"]["platforminfos"] if
                                x["ID"] == 6 or x["ID"] == 19 or x["ID"] == 26]
            obs = self.envs[batch_ix].obs_encoding(self.filt_my_plane, self.enemy_plane)  # TODO 在这里得到了观测值?
            for agent_name, agent_obs in obs.items():
                ret_obs[agent_name][batch_ix] = agent_obs
        return ret_obs

    def initialize_memory(self):
        memory = {}  # this is a map of agent_name -> a list of length self.BATCH_SIZE, but batches that are done are set to None
        for agent in self.AGENT_NAMES:
            memory[agent] = []
            for actual_batch_ix in range(0, self.BATCH_SIZE):
                if not self.DONE_ENVS[actual_batch_ix]:
                    memory[agent].append(Experience())
                else:
                    memory[agent].append(None)
            assert len(memory[agent]) == self.BATCH_SIZE, 'Error Here'
        return memory

    def k_level_communication(self, policies, policy_initial, num_left_batches, left_batches, k_levels):
        for k in range(0, k_levels):
            output_dist = {}
            for agent_ix, agent in enumerate(self.AGENT_NAMES):
                batched_neighbors = [[] for _ in range(0, num_left_batches)]  # for each batch, the policies of agent
                for batch_ix in range(0, num_left_batches):
                    actual_batch_number = left_batches[batch_ix]
                    neighbor_mask_for_agent = (self.adj_matrix[actual_batch_number][agent_ix] == 1)
                    neighbor_names = self.AGENT_NAMES[neighbor_mask_for_agent]
                    for neighbor in neighbor_names:
                        batched_neighbors[batch_ix].append([policy_initial[neighbor][batch_ix], neighbor, batch_ix])
                latent_vector = policies[agent].forward(policy_initial[agent], 1, batched_neighbors)
                output_dist[agent] = latent_vector
            policy_initial = output_dist
        return policy_initial

    def batch_step(self, observations, actions, step_num, time_penalty, early_reward_benefit):
        num_left_batches = np.count_nonzero(
            self.DONE_ENVS == False)  # 得到一个整数类型的变量, 表示self.DONE_ENVS数组中等于False的元素的个数.e.g.4
        left_batches = np.where(self.DONE_ENVS == False)[0]  # e.g.[0 1 2 3]

        next_observations = {agent: np.zeros((num_left_batches,) + self.OBS_SHAPE) for agent in self.AGENT_NAMES}  # 初始化
        rewards = {agent: np.zeros((num_left_batches)) for agent in self.AGENT_NAMES}  # 初始化
        dones = {agent: np.zeros((num_left_batches), dtype=np.bool) for agent in self.AGENT_NAMES}  # 初始化
        for batch_ix in range(0, num_left_batches):
            actual_batch_ix = left_batches[batch_ix]
            my_plane = [planeinfo(plane) for plane in self.filt_my_plane]
            enemy_plane = [planeinfo(plane) for plane in self.enemy_plane]
            self.decision.updateplaneinfo(my_plane, enemy_plane)
            action_dict = {
                agent: self.decision.switchcase(batched_action[batch_ix], planeinfo(self.Name2agent(agent))) for
                agent, batched_action in
                actions.items()}
            # action_dict = {
            #     agent: self.decision.switchcase(0, planeinfo(self.Name2agent(agent))) for agent, batched_action in actions.items()
            # }

            action_list = []
            for i in action_dict.values():
                if i:
                    action_list.append(i[0])
            my_missile = self.obs["red"]["missileinfos"]

            # 蓝方策略和其他飞机自毁
            tmp = self.envs[batch_ix].get_action(self.obs)
            action_list = action_list + tmp
            obs_temp = self.envs[actual_batch_ix].step(action_list)  # TODO step函数可能需要重写
            self.obs = obs_temp
            dones_temp = self.envs[actual_batch_ix].get_done(obs_temp)
            rewards_list = self.reward(my_plane, enemy_plane, my_missile)
            rewards_temp = {}
            for agent, reward in zip(self.AGENT_NAMES, rewards_list):
                rewards_temp[agent] = reward
            # self.update_plane_info()
            # batch_finished = all(dones_temp.values())
            batch_finished = dones_temp[0]
            if batch_finished:
                self.DONE_ENVS[actual_batch_ix] = True

            self.filt_my_plane = [x for x in self.obs["red"]["platforminfos"] if
                                  x["ID"] == 2 or x["ID"] == 11 or x["ID"] == 12 or x["ID"] == 23]
            self.bait_my_plane = [x for x in self.obs["red"]["platforminfos"] if
                                  x["ID"] == 20 or x["ID"] == 21 or x["ID"] == 22]
            self.enemy_plane = [x for x in self.obs["blue"]["platforminfos"] if
                                x["ID"] == 6 or x["ID"] == 19 or x["ID"] == 26]
            self.class_my_plane = [planeinfo(plane) for plane in self.filt_my_plane]
            self.class_bait_plane = [planeinfo(plane) for plane in self.bait_my_plane]
            self.class_enemy_plane = [planeinfo(plane) for plane in self.enemy_plane]

            obs_dict = self.envs[batch_ix].obs_encoding(self.filt_my_plane, self.enemy_plane)

            for agent in self.AGENT_NAMES:
                reward = (rewards_temp[agent] / self.REWARD_SCALE) - abs(time_penalty)
                if self.DONE_ENVS[actual_batch_ix] and step_num < 0.5 * self.MAX_CYCLES:
                    reward = reward + early_reward_benefit
                try:
                    next_observations[agent][batch_ix] = obs_dict[agent]
                except:
                    self.DONE_ENVS[actual_batch_ix] = True
                    return observations, rewards, dones_temp
                rewards[agent][batch_ix] = reward
                # dones[agent][batch_ix] = dones_temp[agent]
        return next_observations, rewards, dones_temp

    def update_plane_info(self):
        pass

    def Name2agent(self, name: str):
        for plane in self.filt_my_plane:
            if name in plane.values():
                return plane

    def add_rewards_to_memory(self, policies, memory, rewards, num_left_batches, left_batches):
        for agent in self.AGENT_NAMES:
            for batch_ix in range(0, num_left_batches):
                actual_batch_number = left_batches[batch_ix]
                memory[agent][actual_batch_number].rewards = rewards[agent][batch_ix]
            policies[agent].add_to_memory(memory[agent])

    def conclude_step(self, next_observations, dones):
        observations = {}
        for agent in self.AGENT_NAMES:
            observations[agent] = next_observations[agent]
        self.update_adj_matrix()
        for agent in observations:
            tmp = torch.tensor(observations[agent]).reshape(self.OBS_SHAPE).to(
                dtype=torch.float32)
            observations[agent] = tmp.clone().detach()
        return observations

    def compute_epoch_data(self, policies, verbose=True, eval=False, standardize_rewards=False):
        epoch_data = {}
        iterations = 0
        for agent in self.AGENT_NAMES:
            loss, batched_mean_reward, batched_length_iteration = policies[agent].compute_loss(
                standardize_rewards=standardize_rewards)
            mean_reward = batched_mean_reward.mean()
            mean_iteration_length = batched_length_iteration.mean()
            iterations = iterations + mean_iteration_length
            epoch_data[agent] = [mean_reward, mean_iteration_length, batched_mean_reward, batched_length_iteration]
            # print('Performing backprop on %s' % (agent))
            if verbose:
                print('\t Reward for %s: %f' % (agent, mean_reward))  # TODO 输出每一代的平均reward
            if not eval:
                loss.backward(retain_graph=True)
        return epoch_data, iterations / self.N_AGENTS

    def conclude_epoch(self, policies, optimizers, schedulers):
        for agent in self.AGENT_NAMES:
            optimizers[agent].step()
            if schedulers is not None:
                schedulers[agent].step()
        self.clear_memory(policies)

    def clear_memory(self, policies):
        for agent in self.AGENT_NAMES:
            policies[agent].clear_memory()

    def consensus_update(self, policies):
        vnet_copies = {agent: policies[agent].policy.v_net.state_dict() for agent in policies.keys()}
        for agent_ix in range(0, self.N_AGENTS):
            neighbor_name_set = set()
            for batch_ix in range(0, self.BATCH_SIZE):
                neighbor_names = self.AGENT_NAMES[self.adj_matrix[batch_ix, agent_ix] == 1]
                for neighbor in neighbor_names:
                    if neighbor not in neighbor_name_set:
                        neighbor_name_set.add(neighbor)
            if len(neighbor_name_set) == 0:
                continue
            neighbor_vnet_copies = [vnet_copies[name] for name in neighbor_name_set]
            policies[self.AGENT_NAMES[agent_ix]].consensus_update(neighbor_vnet_copies)
