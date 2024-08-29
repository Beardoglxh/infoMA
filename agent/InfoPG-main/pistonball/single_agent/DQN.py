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
from reward_1229 import Reward
from plane_decision import Decision
import matplotlib.pyplot as plt
import rl_utils


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

        # print(len(transition_dict['next_states']))
        next_states = torch.tensor(transition_dict['next_states'],
                              dtype=torch.float).to(self.device)
        # print(next_states.shape)
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
        path = "model/" + str(num) + "-" + str(epoch)
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
        self.filt_my_plane = [x for x in self.obs["red"]["platforminfos"] if
                              (x["ID"] == 20 or x["ID"] == 11 or x["ID"] == 22 or x["ID"] == 13)]
        self.bait_my_plane = [x for x in self.obs["red"]["platforminfos"] if (x["ID"] == 2 or x["ID"] == 12 or x["ID"] == 23)]
        self.enemy_plane = [x for x in self.obs["blue"]["platforminfos"] if
                            x["ID"] == 6 or x["ID"] == 19 or x["ID"] == 26]
        self.class_my_plane = [planeinfo(plane) for plane in self.filt_my_plane]
        self.class_bait_my_plane = [planeinfo(plane) for plane in self.bait_my_plane]
        self.class_enemy_plane = [planeinfo(plane) for plane in self.enemy_plane]
        self.reward = Reward(self.class_my_plane, self.class_bait_my_plane, self.class_enemy_plane)
        self.decision = Decision()
        self.done = False

    def batch_reset(self):
        self.done = False
        self.reward.reset()
        self.obs = self.env.system_reset()
        while True:
            if self.obs is None:
                sleep(0.2)
                self.obs = self.env.getObs()
                print("can not get obs now")
                continue
            else:
                break
        self.filt_my_plane = [x for x in self.obs["red"]["platforminfos"] if
                              (x["ID"] == 20 or x["ID"] == 11 or x["ID"] == 22 or x["ID"] == 13)]
        self.bait_my_plane = [x for x in self.obs["red"]["platforminfos"] if (x["ID"] == 2 or x["ID"] == 12 or x["ID"] == 23)]
        self.enemy_plane = [x for x in self.obs["blue"]["platforminfos"] if
                            x["ID"] == 6 or x["ID"] == 19 or x["ID"] == 26]
        raw_obs = self.env.obs_encoding(self.filt_my_plane, self.bait_my_plane, self.enemy_plane).values()
        obs1 = list(raw_obs)[0].squeeze()
        obs2 = list(raw_obs)[1].squeeze()
        obs3 = list(raw_obs)[2].squeeze()
        return obs1, obs2, obs3

    def batch_step(self, actions):
        my_plane = [planeinfo(plane) for plane in self.filt_my_plane]
        bait_my_plane = [planeinfo(plane) for plane in self.bait_my_plane]
        enemy_plane = [planeinfo(plane) for plane in self.enemy_plane]
        self.decision.updateplaneinfo(my_plane, bait_my_plane, enemy_plane)
        action_list = []
        for i in range(len(bait_my_plane)):
            action = self.decision.switchcase(actions[i], bait_my_plane[i])
            action_list.append(action[0])
        my_missile = self.obs["red"]["missileinfos"]

        tmp = self.env.get_action(self.obs)
        action_list = action_list + tmp
        obs_temp = self.env.step(action_list)
        self.obs = obs_temp
        self.filt_my_plane = [x for x in self.obs["red"]["platforminfos"] if
                              (x["ID"] == 20 or x["ID"] == 11 or x["ID"] == 22 or x["ID"] == 13)]
        self.bait_my_plane = [x for x in self.obs["red"]["platforminfos"] if (x["ID"] == 2 or x["ID"] == 12 or x["ID"] == 23)]
        self.enemy_plane = [x for x in self.obs["blue"]["platforminfos"] if
                            x["ID"] == 6 or x["ID"] == 19 or x["ID"] == 26]

        done_temp = self.env.get_done(self.obs)
        done = done_temp[0]
        rewards = self.reward(my_plane, bait_my_plane, enemy_plane).values()
        rewards = list(rewards)
        if done or len(self.bait_my_plane) < 3:
            next_states = []
            next_states.append(torch.zeros(30))
            next_states.append(torch.zeros(30))
            next_states.append(torch.zeros(30))
            return next_states, rewards, done
        raw_obs = self.env.obs_encoding(self.filt_my_plane, self.bait_my_plane, self.enemy_plane).values()
        next_obs = []
        obs1 = list(raw_obs)[0].squeeze()
        # print(len(self.filt_my_plane))
        # print(len(self.bait_my_plane))
        # print(len(self.enemy_plane))
        # print(obs1.size())
        obs2 = list(raw_obs)[1].squeeze()
        obs3 = list(raw_obs)[2].squeeze()
        next_obs.append(obs1)
        next_obs.append(obs2)
        next_obs.append(obs3)
        return next_obs, rewards, done


lr = 2e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'Single-agent'
env = xsimEnv()
random.seed(0)
np.random.seed(0)
# env.seed(0)
torch.manual_seed(0)
replay_buffer1 = ReplayBuffer(buffer_size)
replay_buffer2 = ReplayBuffer(buffer_size)
replay_buffer3 = ReplayBuffer(buffer_size)
state_dim = 30
action_dim = 3
agent1 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)
agent2 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)
agent3 = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)
# agent.load_model()
agent1.load_model("model/1-16")
agent2.load_model("model/2-6")
agent3.load_model("model/3-6")

return_list1 = []
return_list2 = []
return_list3 = []

for i in range(400):
    with tqdm(total=int(num_episodes / 400), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 400)):
            # replay_buffer = ReplayBuffer(buffer_size)
            episode_return1 = 0
            episode_return2 = 0
            episode_return3 = 0
            state1, state2, state3 = env.batch_reset()
            done = False
            while not done:
                actions = []
                action1 = agent1.take_action(state1)
                action2 = agent1.take_action(state2)
                action3 = agent1.take_action(state3)
                actions.append(action1)
                actions.append(action2)
                actions.append(action3)
                next_states, rewards, done = env.batch_step(actions)

                replay_buffer1.add(state1, action1, rewards[0], next_states[0], done)
                replay_buffer2.add(state2, action2, rewards[1], next_states[1], done)
                replay_buffer3.add(state3, action3, rewards[2], next_states[2], done)
                state1 = next_states[0]
                state2 = next_states[1]
                state3 = next_states[2]

                episode_return1 += rewards[0]
                episode_return2 += rewards[1]
                episode_return3 += rewards[2]
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer1.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer1.sample(batch_size)
                    transition_dict1 = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent1.update(transition_dict1)

                if replay_buffer2.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer2.sample(batch_size)
                    transition_dict2 = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent2.update(transition_dict2)

                if replay_buffer3.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer1.sample(batch_size)
                    transition_dict3 = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent3.update(transition_dict3)

            return_list1.append(episode_return1)
            return_list2.append(episode_return2)
            return_list3.append(episode_return3)

            if (i_episode + 1) % 5 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 400 * i + i_episode + 1),
                    'return1':
                        '%.3f' % np.mean(return_list1[-5:]),
                    'return2':
                        '%.3f' % np.mean(return_list2[-5:]),
                    'return3':
                        '%.3f' % np.mean(return_list3[-5:]),
                })
            pbar.update(1)
    agent1.save_model(1,i)
    agent2.save_model(2,i)
    agent3.save_model(3,i)

# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('DQN on {}'.format(env_name))
# plt.show()
#
# mv_return = rl_utils.moving_average(return_list, 9)
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('DQN on {}'.format(env_name))
# plt.show()
