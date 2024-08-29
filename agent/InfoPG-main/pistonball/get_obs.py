from time import sleep
from multiprocessing import Pool
import torch
from config import ADDRESS, config, ISHOST, XSIM_NUM
from env.env_runner import EnvRunner


class startXsim(EnvRunner):
    def __init__(self, address: str):
        EnvRunner.__init__(self, address)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print('**Using: ', torch.cuda.get_device_name(device))
        else:
            device = torch.device("cpu")
            print('**Using: cpu')

        self.device = device

    def getObs(self):
        obs = self.step([])
        return obs

    def bait_obs_encoding(self, filt_plane, bait_plane, enemy_plane):
        obervations = []
        enemy_info = torch.tensor([[]]).to(self.device)
        filt_info = torch.tensor([[]]).to(self.device)

        for plane in enemy_plane:
            obervation = []
            if plane["Availability"] == 1:
                obervation.append(plane["X"] / 1e+3)
                obervation.append(plane["Y"] / 1e+3)
                obervation = torch.tensor(obervation).unsqueeze(dim=0).to(self.device)
            enemy_info = torch.cat((enemy_info, obervation), dim=1)

        if len(enemy_plane) < 3:
            tmp = torch.zeros((1, 2 * (3 - len(enemy_plane)))).to(self.device)
            enemy_info = torch.cat((enemy_info, tmp), dim=1)

        for plane in filt_plane:
            obervation = []
            if plane["Availability"] == 1:
                obervation.append(plane["X"] / 1e+3)
                obervation.append(plane["Y"] / 1e+3)
                obervation = torch.tensor(obervation).unsqueeze(dim=0).to(self.device)
            filt_info = torch.cat((filt_info, obervation), dim=1)

        if len(filt_plane) < 4:
            tmp = torch.zeros((1, 2 * (4 - len(filt_plane)))).to(self.device)
            filt_info = torch.cat((filt_info, tmp), dim=1)

        for plane in bait_plane:
            obervation = []
            if plane["Availability"] == 1:
                obervation.append(plane["X"] / 1e+3)
                obervation.append(plane["Y"] / 1e+3)
                obervation = torch.tensor(obervation).unsqueeze(dim=0).to(self.device)
                obervation = torch.cat((obervation, enemy_info), dim=1)
                obervation = torch.cat((obervation, filt_info), dim=1)
            obervations.append(obervation)

        agent_names = [x["Name"] for x in bait_plane]
        last = {}
        for agent_name, obs in zip(agent_names, obervations):
            last[agent_name] = obs
        return last


    def filt_obs_encoding(self, filt_my_plane, enemy_plane):
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
                obervation = torch.tensor(obervation).unsqueeze(dim=0).to(self.device)
            enemy_info = torch.cat((enemy_info, obervation), dim=1)

        if len(enemy_plane) < 3:
            tmp = torch.zeros((1, 5 * (3 - len(enemy_plane)))).to(self.device)
            enemy_info = torch.cat((enemy_info, tmp), dim=1)

        for plane in filt_my_plane:
            obervation = []
            if plane["Availability"] == 1:
                obervation.append(plane["Heading"])
                obervation.append(plane["Speed"] / 1e+2)
                obervation.append(plane["X"] / 1e+3)
                obervation.append(plane["Y"] / 1e+3)
                obervation.append(plane["Alt"] / 1e+3)
                obervation = torch.tensor(obervation).unsqueeze(dim=0).to(self.device)
                obervation = torch.cat((obervation, enemy_info), dim=1)
            obervations.append(obervation)
        agent_names = [x["Name"] for x in filt_my_plane]
        last = {}
        for agent_name, obs in zip(agent_names, obervations):
            last[agent_name] = obs
        return last

    def enemy_obs_encoding(self, filt_my_plane, enemy_plane):
        obervations = []
        filt_info = torch.tensor([[]]).to(self.device)

        for plane in filt_my_plane:
            obervation = []
            if plane["Availability"] == 1:
                obervation.append(plane["Heading"])
                obervation.append(plane["Speed"] / 1e+2)
                obervation.append(plane["X"] / 1e+3)
                obervation.append(plane["Y"] / 1e+3)
                obervation.append(plane["Alt"] / 1e+3)
                obervation = torch.tensor(obervation).unsqueeze(dim=0).to(self.device)
            filt_info = torch.cat((filt_info, obervation), dim=1)

        if len(filt_my_plane) < 4:
            tmp = torch.zeros((1, 5 * (4 - len(filt_my_plane)))).to(self.device)
            filt_info = torch.cat((filt_info, tmp), dim=1)

        for plane in enemy_plane:
            obervation = []
            if plane["Availability"] == 1:
                obervation.append(plane["Heading"])
                obervation.append(plane["Speed"] / 1e+2)
                obervation.append(plane["X"] / 1e+3)
                obervation.append(plane["Y"] / 1e+3)
                obervation.append(plane["Alt"] / 1e+3)
                obervation = torch.tensor(obervation).unsqueeze(dim=0).to(self.device)
                obervation = torch.cat((obervation, filt_info), dim=1)
            obervations.append(obervation)
        agent_names = [x["Name"] for x in enemy_plane]
        last = {}
        for agent_name, obs in zip(agent_names, obervations):
            last[agent_name] = obs
        return last

    def system_reset(self):
        # 智能体重置
        # self.__init_agents()
        self.system__reset_agents()
        # 环境重置
        self.reset()
        obs = self.step([])
        # while obs["sim_time"] > 10:
        #     obs = self.step([])
        return obs

    def get_action(self, obs):
        actions = []
        cur_time = obs["sim_time"]
        global_obs = obs
        cmd_list = []
        for side, agent in self.agents.items():
            if side == "red":
                cmd_list = self._agent_step(agent, cur_time, obs[side], global_obs)
            else:
                cmd_list = self._agent_step(agent, cur_time, obs[side], global_obs)
            # print(cmd_list)
            actions.extend(cmd_list)

        return actions

    def _agent_step(self, agent, cur_time, obs_side, global_obs):
        cmd_list = agent.step(cur_time, obs_side, global_obs)
        return cmd_list

# address = ADDRESS['ip'] + ":" + str(ADDRESS['port'])
# a = startXsim(address)
# b = a.getObs()
# print(b)
