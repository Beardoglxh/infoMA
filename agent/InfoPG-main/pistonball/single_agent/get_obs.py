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

    def obs_encoding(self, filt_my_plane, bait_my_plane, enemy_plane):
        obervations = []

        #敌机信息
        enemy_info1 = torch.tensor([[]]).to(self.device)
        enemy_info2 = torch.tensor([[]]).to(self.device)
        enemy_info3 = torch.tensor([[]]).to(self.device)
        obervation1 = []
        obervation2 = []
        obervation3 = []

        obervation1.append(enemy_plane[0]["Heading"])
        obervation1.append(enemy_plane[0]["Speed"] / 1e+2)
        obervation1.append(enemy_plane[0]["X"] / 1e+3)
        obervation1.append(enemy_plane[0]["Y"] / 1e+3)
        obervation1.append(enemy_plane[0]["Alt"] / 1e+3)

        obervation2.append(enemy_plane[1]["Heading"])
        obervation2.append(enemy_plane[1]["Speed"] / 1e+2)
        obervation2.append(enemy_plane[1]["X"] / 1e+3)
        obervation2.append(enemy_plane[1]["Y"] / 1e+3)
        obervation2.append(enemy_plane[1]["Alt"] / 1e+3)

        obervation3.append(enemy_plane[2]["Heading"])
        obervation3.append(enemy_plane[2]["Speed"] / 1e+2)
        obervation3.append(enemy_plane[2]["X"] / 1e+3)
        obervation3.append(enemy_plane[2]["Y"] / 1e+3)
        obervation3.append(enemy_plane[2]["Alt"] / 1e+3)

        obervation1 = torch.tensor(obervation1).unsqueeze(dim=0).to(self.device)
        obervation2 = torch.tensor(obervation2).unsqueeze(dim=0).to(self.device)
        obervation3 = torch.tensor(obervation3).unsqueeze(dim=0).to(self.device)
        enemy_info1 = torch.cat((enemy_info1, obervation1), dim=1)
        enemy_info2 = torch.cat((enemy_info2, obervation2), dim=1)
        enemy_info3 = torch.cat((enemy_info3, obervation3), dim=1)

        # 我机信息
        my_plane_info = torch.tensor([[]]).to(self.device)
        for plane in filt_my_plane:
            obervation = []
            if plane["Availability"] == 1:
                obervation.append(plane["Heading"])
                obervation.append(plane["Speed"] / 1e+2)
                obervation.append(plane["X"] / 1e+3)
                obervation.append(plane["Y"] / 1e+3)
                obervation.append(plane["Alt"] / 1e+3)
                obervation = torch.tensor(obervation).unsqueeze(dim=0).to(self.device)
            my_plane_info = torch.cat((my_plane_info, obervation), dim=1)

        # 诱饵机信息
        obervation1 = []
        obervation2 = []
        obervation3 = []

        obervation1.append(bait_my_plane[0]["Heading"])
        obervation1.append(bait_my_plane[0]["Speed"] / 1e+2)
        obervation1.append(bait_my_plane[0]["X"] / 1e+3)
        obervation1.append(bait_my_plane[0]["Y"] / 1e+3)
        obervation1.append(bait_my_plane[0]["Alt"] / 1e+3)

        obervation2.append(bait_my_plane[1]["Heading"])
        obervation2.append(bait_my_plane[1]["Speed"] / 1e+2)
        obervation2.append(bait_my_plane[1]["X"] / 1e+3)
        obervation2.append(bait_my_plane[1]["Y"] / 1e+3)
        obervation2.append(bait_my_plane[1]["Alt"] / 1e+3)

        obervation3.append(bait_my_plane[2]["Heading"])
        obervation3.append(bait_my_plane[2]["Speed"] / 1e+2)
        obervation3.append(bait_my_plane[2]["X"] / 1e+3)
        obervation3.append(bait_my_plane[2]["Y"] / 1e+3)
        obervation3.append(bait_my_plane[2]["Alt"] / 1e+3)

        obervation1 = torch.tensor(obervation1).unsqueeze(dim=0).to(self.device)
        obervation2 = torch.tensor(obervation2).unsqueeze(dim=0).to(self.device)
        obervation3 = torch.tensor(obervation3).unsqueeze(dim=0).to(self.device)
        obervation1 = torch.cat((obervation1, enemy_info1), dim=1)
        obervation2 = torch.cat((obervation2, enemy_info2), dim=1)
        obervation3 = torch.cat((obervation3, enemy_info3), dim=1)
        obervation1 = torch.cat((obervation1, my_plane_info), dim=1)
        obervation2 = torch.cat((obervation2, my_plane_info), dim=1)
        obervation3 = torch.cat((obervation3, my_plane_info), dim=1)

        obervations.append(obervation1)
        obervations.append(obervation2)
        obervations.append(obervation3)


        agent_names = [x["Name"] for x in bait_my_plane]
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
