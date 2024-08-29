import datetime
import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
from xsim_config import address
from get_obs import startXsim
from time import sleep
from reward_1229 import Bait_Reward
from plane_decision import Decision
import matplotlib.pyplot as plt
import rl_utils
from policy_piston import PistonPolicy
import torch.optim as optim
from policy_base import Experience
from torch.distributions import Categorical
import os
import json
import csv


class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        state_np = tuple(t.numpy() for t in state)
        next_state_np = tuple(t.numpy() for t in next_state)
        return state_np, action, reward, next_state_np, done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    """ 只有一层隐藏层的Q网络 """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    """ DQN算法 """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)

        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)

        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save_model(self, num, epoch):
        path = "DQNmodel/" + str(num) + "-" + str(epoch)
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, name):
        self.q_net.load_state_dict(torch.load(name))


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


class xsimEnv:
    def __init__(self):
        print("Try to create xsim")
        self.address = address["ip"] + ":" + str(address["port"])
        self.env = startXsim(self.address)
        self.obs = self.env.getObs()
        self.DY_1 = [x for x in self.obs["red"]["platforminfos"] if
                     x["ID"] == 41 or x["ID"] == 42 or x["ID"] == 43 or x["ID"] == 44]
        self.DY_2 = [x for x in self.obs["red"]["platforminfos"] if
                     x["ID"] == 45 or x["ID"] == 46 or x["ID"] == 47 or x["ID"] == 48]
        self.H4 = [x for x in self.obs["red"]["platforminfos"] if
                   x["ID"] == 55 or x["ID"] == 56 or x["ID"] == 57 or x["ID"] == 58]
        self.H5 = [x for x in self.obs["red"]["platforminfos"] if
                   x["ID"] == 49 or x["ID"] == 50 or x["ID"] == 51 or x["ID"] == 52]
        self.YE_1 = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 35 or x["ID"] == 36 or x["ID"] == 37]
        self.YE_2 = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 38 or x["ID"] == 39 or x["ID"] == 40]
        self.YJ = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 53 or x["ID"] == 54]
        self.enemy1 = [x for x in self.obs["blue"]["platforminfos"] if x["ID"] == 6 or x["ID"] == 30 or x["ID"] == 33]
        self.enemy2 = [x for x in self.obs["blue"]["platforminfos"] if x["ID"] == 31 or x["ID"] == 32 or x["ID"] == 34]
        self.data = dict()
        self.single_data = {
            "YE1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE3": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE4": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE5": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE6": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY3": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY4": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY5": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY6": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY7": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY8": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H41": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H42": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H43": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H44": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YJ1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YJ2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H51": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H52": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H53": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H54": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA3": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA4": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "B211": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "B212": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
        }
        self.decision = Decision()
        self.done = False

    def batch_reset(self):
        self.done = False
        self.obs = self.env.system_reset()
        while True:
            if self.obs is None:
                sleep(0.2)
                self.obs = self.env.getObs()
                print("can not get obs now")
                continue
            else:
                break
        self.DY_1 = [x for x in self.obs["red"]["platforminfos"] if
                     x["ID"] == 41 or x["ID"] == 42 or x["ID"] == 43 or x["ID"] == 44]
        self.DY_2 = [x for x in self.obs["red"]["platforminfos"] if
                     x["ID"] == 45 or x["ID"] == 46 or x["ID"] == 47 or x["ID"] == 48]
        self.H4 = [x for x in self.obs["red"]["platforminfos"] if
                   x["ID"] == 55 or x["ID"] == 56 or x["ID"] == 57 or x["ID"] == 58]
        self.H5 = [x for x in self.obs["red"]["platforminfos"] if
                   x["ID"] == 49 or x["ID"] == 50 or x["ID"] == 51 or x["ID"] == 52]
        self.YE_1 = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 35 or x["ID"] == 36 or x["ID"] == 37]
        self.YE_2 = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 38 or x["ID"] == 39 or x["ID"] == 40]
        self.YJ = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 53 or x["ID"] == 54]
        self.enemy1 = [x for x in self.obs["blue"]["platforminfos"] if x["ID"] == 6 or x["ID"] == 30 or x["ID"] == 33]
        self.enemy2 = [x for x in self.obs["blue"]["platforminfos"] if x["ID"] == 31 or x["ID"] == 32 or x["ID"] == 34]
        self.single_data = {
            "YE1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE3": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE4": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE5": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE6": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY3": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY4": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY5": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY6": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY7": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY8": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H41": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H42": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H43": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H44": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YJ1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YJ2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H51": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H52": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H53": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H54": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA3": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA4": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "B211": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "B212": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
        }
        for value in self.obs.values():
            if type(value) != float:
                for plane in value["platforminfos"]:
                    self.single_data[plane["Name"]] = {"ID": plane["ID"], "X": plane["X"], "Y": plane["Y"],
                                                       "Z": plane["Alt"], "is_alive": True}
            else:
                self.data[value] = self.single_data
                break

        dy1_obs = self.env.filt_obs_encoding(self.DY_1, self.enemy1)
        dy2_obs = self.env.filt_obs_encoding(self.DY_2, self.enemy2)
        h4_obs = self.env.filt_obs_encoding(self.H4, self.enemy1)
        h5_obs = self.env.filt_obs_encoding(self.H5, self.enemy2)
        ye1_obs = self.env.filt_obs_encoding(self.YE_1, self.enemy1)
        ye2_obs = self.env.filt_obs_encoding(self.YE_2, self.enemy2)
        enemy1_obs = self.env.enemy_obs_encoding(self.DY_1, self.enemy1)
        enemy2_obs = self.env.enemy_obs_encoding(self.DY_2, self.enemy2)

        return dy1_obs, dy2_obs, h4_obs, h5_obs, ye1_obs, ye2_obs, enemy1_obs, enemy2_obs

    def batch_step(self, DY1_actions, DY2_actions, H4_actions, H5_actions, YE1_actions, YE2_actions, ENEMY1_actions,
                   ENEMY2_actions):
        DY_1 = [planeinfo(plane) for plane in self.DY_1]
        DY_2 = [planeinfo(plane) for plane in self.DY_2]
        H4 = [planeinfo(plane) for plane in self.H4]
        H5 = [planeinfo(plane) for plane in self.H5]
        YE1 = [planeinfo(plane) for plane in self.YE_1]
        YE2 = [planeinfo(plane) for plane in self.YE_2]
        enemy1 = [planeinfo(plane) for plane in env.enemy1]
        enemy2 = [planeinfo(plane) for plane in env.enemy2]
        # self.decision.updateplaneinfo(DY_1, YE1, enemy1)

        action_list = []
        self.get_cmd(DY_1, DY1_actions, action_list)
        self.get_cmd(DY_2, DY2_actions, action_list)
        self.get_cmd(H4, H4_actions, action_list)
        self.get_cmd(H5, H5_actions, action_list)
        self.get_cmd(YE1, YE1_actions, action_list)
        self.get_cmd(YE2, YE2_actions, action_list)
        self.get_cmd(enemy1, ENEMY1_actions, action_list)
        self.get_cmd(enemy2, ENEMY2_actions, action_list)

        tmp = self.env.get_action(self.obs)
        action_list = action_list + tmp

        self.obs = self.env.step(action_list)
        while True:
            if self.obs is None:
                sleep(0.2)
                self.obs = self.env.step(action_list)
                print("can not get obs now")
                continue
            else:
                break
        self.single_data = {
            "YE1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE3": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE4": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE5": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YE6": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY3": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY4": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY5": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY6": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY7": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "DY8": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H41": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H42": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H43": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H44": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YJ1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "YJ2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H51": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H52": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H53": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "H54": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA1": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA2": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA3": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "PCA4": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "B211": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
            "B212": {"ID": None, "X": None, "Y": None, "Z": None, "is_alive": False},
        }
        for value in self.obs.values():
            if type(value) != float:
                for plane in value["platforminfos"]:
                    self.single_data[plane["Name"]] = {"ID": plane["ID"], "X": plane["X"], "Y": plane["Y"],
                                                       "Z": plane["Alt"], "is_alive": True}
            else:
                self.data[value] = self.single_data
                break
        self.DY_1 = [x for x in self.obs["red"]["platforminfos"] if
                     x["ID"] == 41 or x["ID"] == 42 or x["ID"] == 43 or x["ID"] == 44]
        self.DY_2 = [x for x in self.obs["red"]["platforminfos"] if
                     x["ID"] == 45 or x["ID"] == 46 or x["ID"] == 47 or x["ID"] == 48]
        self.H4 = [x for x in self.obs["red"]["platforminfos"] if
                   x["ID"] == 55 or x["ID"] == 56 or x["ID"] == 57 or x["ID"] == 58]
        self.H5 = [x for x in self.obs["red"]["platforminfos"] if
                   x["ID"] == 49 or x["ID"] == 50 or x["ID"] == 51 or x["ID"] == 52]
        self.YE_1 = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 35 or x["ID"] == 36 or x["ID"] == 37]
        self.YE_2 = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 38 or x["ID"] == 39 or x["ID"] == 40]
        self.YJ = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 53 or x["ID"] == 54]
        self.enemy1 = [x for x in self.obs["blue"]["platforminfos"] if x["ID"] == 6 or x["ID"] == 30 or x["ID"] == 33]
        self.enemy2 = [x for x in self.obs["blue"]["platforminfos"] if x["ID"] == 31 or x["ID"] == 32 or x["ID"] == 34]
        self.done = True
        for x in self.enemy1:
            if x["ID"] == 33:
                self.done = False
        for x in self.enemy2:
            if x["ID"] == 34:
                self.done = False

        dy1_obs = self.env.filt_obs_encoding(self.DY_1, self.enemy1)
        dy2_obs = self.env.filt_obs_encoding(self.DY_2, self.enemy2)
        h4_obs = self.env.filt_obs_encoding(self.H4, self.enemy1)
        h5_obs = self.env.filt_obs_encoding(self.H5, self.enemy2)
        ye1_obs = self.env.filt_obs_encoding(self.YE_1, self.enemy1)
        ye2_obs = self.env.filt_obs_encoding(self.YE_2, self.enemy2)
        enemy1_obs = self.env.enemy_obs_encoding(self.DY_1, self.enemy1)
        enemy2_obs = self.env.enemy_obs_encoding(self.DY_2, self.enemy2)
        # ye1_obs = self.env.bait_obs_encoding(self.DY_1, self.YE_1, self.enemy1)
        # ye2_obs = self.env.bait_obs_encoding(self.DY_2, self.YE_2, self.enemy2)
        return dy1_obs, dy2_obs, h4_obs, h5_obs, ye1_obs, ye2_obs, enemy1_obs, enemy2_obs

    def get_agent_names(self, planes):
        AGENT_NAMES = np.array([x["Name"] for x in planes])
        return AGENT_NAMES

    def get_cmd(self, Planes, actions, action_list):
        for i in range(len(Planes)):
            action = self.decision.switchcase(actions[i], Planes[i])
            if action is None:
                continue
            action_list.append(action[0])
        return action_list


# 需要修改
def k_level_communication(policies, policy_initial, num_left_batches, left_batches, k_levels, AGENT_NAMES, adj_matrix):
    for k in range(0, k_levels):
        output_dist = {}
        for agent_ix, agent in enumerate(AGENT_NAMES):
            batched_neighbors = [[] for _ in range(0, num_left_batches)]  # for each batch, the policies of agent
            for batch_ix in range(0, num_left_batches):
                actual_batch_number = left_batches[batch_ix]
                neighbor_mask_for_agent = (adj_matrix[actual_batch_number][agent_ix] == 1)
                neighbor_names = AGENT_NAMES[neighbor_mask_for_agent]
                for neighbor in neighbor_names:
                    batched_neighbors[batch_ix].append([policy_initial[neighbor][batch_ix], neighbor, batch_ix])
            latent_vector = policies[agent].forward(policy_initial[agent], 1, batched_neighbors)
            output_dist[agent] = latent_vector
        policy_initial = output_dist
    return policy_initial


def count_matrix(AGENT_NAMES):
    adj_matrix = np.zeros((1, len(AGENT_NAMES), len(AGENT_NAMES)))
    for i in range(0, 1):
        for j in range(0, len(AGENT_NAMES)):
            if j - 1 >= 0:
                adj_matrix[i][j][j - 1] = 1
            if j + 1 < len(AGENT_NAMES):
                adj_matrix[i][j][j + 1] = 1
    return adj_matrix


def Loadmodel(AGENT_NAMES, path, polices):
    if (path == 'ENEMY1' or path == 'ENEMY2'):
        for agent_name in AGENT_NAMES:
            data = torch.load(os.path.join('experiments', 'pistonball', path, '%s.pt' % (agent_name)),
                              device)
            polices[agent_name] = PistonPolicy(encoding_size_enemy, policy_latent_size, action_space, device,
                                               'normal', model_state_dict=data['policy'])
    else:
        if (len(AGENT_NAMES) == 4):
            for agent_name in AGENT_NAMES:
                data = torch.load(os.path.join('experiments', 'pistonball', path, '%s.pt' % (agent_name)),
                                  device)
                polices[agent_name] = PistonPolicy(encoding_size, policy_latent_size, action_space, device,
                                                   'normal', model_state_dict=data['policy'])
        elif (len(AGENT_NAMES) == 3):
            for agent_name in AGENT_NAMES:
                data = torch.load(os.path.join('experiments', 'pistonball', path, '%s.pt' % (agent_name)),
                                  device)
                polices[agent_name] = PistonPolicy(encoding_size, policy_latent_size, action_space, device,
                                                   'normal', model_state_dict=data['policy'])
        elif (len(AGENT_NAMES) == 2):
            pass


def get_actions(AGENT_NAMES, policies, obs, adj_matrix):
    actions = []
    policy_initial = {}
    for agent in AGENT_NAMES:
        initial_policy_distribution, state_val = policies[agent].forward(obs[agent], 0, None)  # TODO (4, 4096)->(4, 20)
        policy_initial[agent] = initial_policy_distribution

    if communicate:  # TODO 初始策略进行通讯
        policy_initial = k_level_communication(policies, policy_initial, 1, [0], k_levels, AGENT_NAMES, adj_matrix)

    for agent in AGENT_NAMES:
        final_policy_distribution = policies[agent].forward(policy_initial[agent], 2, None).to(
            'cpu').clone()  # TODO (4, 20)->(4, 3) softmax
        distribution = Categorical(probs=final_policy_distribution)
        batch_action = distribution.sample()
        actions.append(batch_action.item())
    return actions


epochs = 2000
lr = 0.001
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# infoPG参数
action_space = 5
encoding_size = 20
encoding_size_enemy = 25
policy_latent_size = 20
n_agents = 4
max_cycles = 20 * 60
max_grad_norm = 0.75
communicate = True
time_penalty = 0.007
early_reward_benefit = 0.85
adv_type = 'clamped_q'
consensus_update = False
k_levels = 1
verbose = True

# DQN参数
num_episodes = 2000
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 300
batch_size = 64
state_dim = 16
action_dim = 3
env_name = 'Single-agent'

env = xsimEnv()
# 随机种子
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# 定义模型
policies_DY_1 = {agent: PistonPolicy(encoding_size, policy_latent_size, action_space, device, adv_type) for agent in
                 env.get_agent_names(env.DY_1)}
policies_DY_2 = {agent: PistonPolicy(encoding_size, policy_latent_size, action_space, device, adv_type) for agent in
                 env.get_agent_names(env.DY_2)}
policies_H4 = {agent: PistonPolicy(encoding_size, policy_latent_size, action_space, device, adv_type) for agent in
               env.get_agent_names(env.H4)}
policies_H5 = {agent: PistonPolicy(encoding_size, policy_latent_size, action_space, device, adv_type) for agent in
               env.get_agent_names(env.H5)}

policies_YE1 = {agent: PistonPolicy(encoding_size, policy_latent_size, action_space, device, adv_type) for agent in
                env.get_agent_names(env.YE_1)}
policies_YE2 = {agent: PistonPolicy(encoding_size, policy_latent_size, action_space, device, adv_type) for agent in
                env.get_agent_names(env.YE_2)}

policies_ENEMY1 = {agent: PistonPolicy(encoding_size_enemy, policy_latent_size, action_space, device, adv_type) for
                   agent in
                   env.get_agent_names(env.enemy1)}
policies_ENEMY2 = {agent: PistonPolicy(encoding_size_enemy, policy_latent_size, action_space, device, adv_type) for
                   agent in
                   env.get_agent_names(env.enemy2)}

DY_1_AGENT_NAMES = env.get_agent_names(env.DY_1)
DY_2_AGENT_NAMES = env.get_agent_names(env.DY_2)
H4_AGENT_NAMES = env.get_agent_names(env.H4)
H5_AGENT_NAMES = env.get_agent_names(env.H5)
YE1_AGENT_NAMES = env.get_agent_names(env.YE_1)
YE2_AGENT_NAMES = env.get_agent_names(env.YE_2)
ENEMY1_AGENT_NAMES = env.get_agent_names(env.enemy1)
ENEMY2_AGENT_NAMES = env.get_agent_names(env.enemy2)

Loadmodel(DY_1_AGENT_NAMES, "DY1", policies_DY_1)
Loadmodel(DY_2_AGENT_NAMES, "DY2", policies_DY_2)
Loadmodel(H4_AGENT_NAMES, "H4", policies_H4)
Loadmodel(H5_AGENT_NAMES, "H5", policies_H5)
Loadmodel(YE1_AGENT_NAMES, "YE1", policies_YE1)
Loadmodel(YE2_AGENT_NAMES, "YE2", policies_YE2)
Loadmodel(ENEMY1_AGENT_NAMES, "ENEMY1", policies_ENEMY1)
Loadmodel(ENEMY2_AGENT_NAMES, "ENEMY2", policies_ENEMY2)

# agent1 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
# agent2 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
# agent3 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
# agent4 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
# agent5 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
# agent6 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
# agent1.load_model("DQNmodel/1")
# agent2.load_model("DQNmodel/2")
# agent3.load_model("DQNmodel/3")
# agent4.load_model("DQNmodel/4")
# agent5.load_model("DQNmodel/5")
# agent6.load_model("DQNmodel/6")

for epoch in range(0, 1):
    cycle = 0

    dy1_obs, dy2_obs, h4_obs, h5_obs, ye1_obs, ye2_obs, enemy1_obs, enemy2_obs = env.batch_reset()

    # DY_1_AGENT_NAMES = env.get_agent_names(env.DY_1)
    # DY_2_AGENT_NAMES = env.get_agent_names(env.DY_2)
    # H4_AGENT_NAMES = env.get_agent_names(env.H4)
    # H5_AGENT_NAMES = env.get_agent_names(env.H5)
    # YE1_AGENT_NAMES = env.get_agent_names(env.YE_1)
    # YE2_AGENT_NAMES = env.get_agent_names(env.YE_2)

    while not env.done:
        DY1_actions = []
        DY2_actions = []
        H4_actions = []
        H5_actions = []
        YE1_actions = []
        YE2_actions = []
        ENEMY1_actions = []
        ENEMY2_actions = []

        DY1_adj_matrix = count_matrix(DY_1_AGENT_NAMES)
        DY2_adj_matrix = count_matrix(DY_2_AGENT_NAMES)
        H4_adj_matrix = count_matrix(H4_AGENT_NAMES)
        H5_adj_matrix = count_matrix(H5_AGENT_NAMES)
        YE1_adj_matrix = count_matrix(YE1_AGENT_NAMES)
        YE2_adj_matrix = count_matrix(YE2_AGENT_NAMES)
        ENEMY1_adj_matrix = count_matrix(ENEMY1_AGENT_NAMES)
        ENEMY2_adj_matrix = count_matrix(ENEMY2_AGENT_NAMES)

        # Loadmodel(DY_1_AGENT_NAMES, "DY1", policies_DY_1)
        # Loadmodel(DY_2_AGENT_NAMES, "DY2", policies_DY_2)
        # Loadmodel(H4_AGENT_NAMES, "H4", policies_H4)
        # Loadmodel(H5_AGENT_NAMES, "H5", policies_H5)
        # Loadmodel(YE1_AGENT_NAMES, "YE1", policies_YE1)
        # Loadmodel(YE2_AGENT_NAMES, "YE2", policies_YE2)

        DY1_actions = get_actions(DY_1_AGENT_NAMES, policies_DY_1, dy1_obs, DY1_adj_matrix)
        DY2_actions = get_actions(DY_2_AGENT_NAMES, policies_DY_2, dy2_obs, DY2_adj_matrix)
        H4_actions = get_actions(H4_AGENT_NAMES, policies_H4, h4_obs, H4_adj_matrix)
        H5_actions = get_actions(H5_AGENT_NAMES, policies_H5, h5_obs, H5_adj_matrix)
        YE1_actions = get_actions(YE1_AGENT_NAMES, policies_YE1, ye1_obs, YE1_adj_matrix)
        YE2_actions = get_actions(YE2_AGENT_NAMES, policies_YE2, ye2_obs, YE2_adj_matrix)
        ENEMY1_actions = get_actions(ENEMY1_AGENT_NAMES, policies_ENEMY1, enemy1_obs, ENEMY1_adj_matrix)
        ENEMY2_actions = get_actions(ENEMY2_AGENT_NAMES, policies_ENEMY2, enemy2_obs, ENEMY2_adj_matrix)

        if cycle < 100:
            DY1_actions = [0, 0, 0, 0]
            DY2_actions = [0, 0, 0, 0]
            H4_actions = [0, 0, 0, 0]
            H5_actions = [0, 0, 0, 0]
            YE1_actions = [0, 0, 0]
            YE2_actions = [0, 0, 0]
            ENEMY1_actions = [0, 0, 0]
            ENEMY2_actions = [0, 0, 0]

        # for agent in env.YE_1:
        #     if (agent["ID"] == 24):
        #         state1 = ye1_obs["YE1"][0]
        #         action1 = agent1.take_action(state1)
        #         YE1_actions.append(action1)
        #     elif (agent["ID"] == 25):
        #         state2 = ye1_obs["YE2"][0]
        #         action2 = agent1.take_action(state2)
        #         YE1_actions.append(action2)
        #     elif (agent["ID"] == 28):
        #         state3 = ye1_obs["YE5"][0]
        #         action3 = agent1.take_action(state3)
        #         YE1_actions.append(action3)
        #
        # for agent in env.YE_2:
        #     if (agent["ID"] == 26):
        #         state4 = ye2_obs["YE3"][0]
        #         action4 = agent1.take_action(state4)
        #         YE2_actions.append(action4)
        #     elif (agent["ID"] == 27):
        #         state5 = ye2_obs["YE4"][0]
        #         action5 = agent1.take_action(state5)
        #         YE2_actions.append(action5)
        #     elif (agent["ID"] == 29):
        #         state6 = ye2_obs["YE6"][0]
        #         action6 = agent1.take_action(state6)
        #         YE2_actions.append(action6)

        dy1_obs, dy2_obs, h4_obs, h5_obs, ye1_obs, ye2_obs, enemy1_obs, enemy2_obs = env.batch_step(DY1_actions,
                                                                                                    DY2_actions,
                                                                                                    H4_actions,
                                                                                                    H5_actions,
                                                                                                    YE1_actions,
                                                                                                    YE2_actions,
                                                                                                    ENEMY1_actions,
                                                                                                    ENEMY2_actions)

        DY_1_AGENT_NAMES = env.get_agent_names(env.DY_1)
        DY_2_AGENT_NAMES = env.get_agent_names(env.DY_2)
        H4_AGENT_NAMES = env.get_agent_names(env.H4)
        H5_AGENT_NAMES = env.get_agent_names(env.H5)
        YE1_AGENT_NAMES = env.get_agent_names(env.YE_1)
        YE2_AGENT_NAMES = env.get_agent_names(env.YE_2)
        ENEMY1_AGENT_NAMES = env.get_agent_names(env.enemy1)
        ENEMY2_AGENT_NAMES = env.get_agent_names(env.enemy2)

        cycle = cycle + 1
        if cycle >= max_cycles:
            done = 1
