from batch_envs import BatchEnv
from typing import Dict
import torch
from policy_piston import PistonPolicy, MOAPolicy
from alex_net import Encoder
import numpy as np
from torch.distributions import Categorical
import torch.optim as optim
from get_obs import startXsim
from config import ADDRESS
import gym
import os
import math

from config import config
from env.xsim_env import XSimEnv


class XsimEnv:
    def __init__(self, batch: int, env_params: Dict, seed=None):
        print("Try Xsim")
        self.BATCH_SIZE = batch
        self.MAX_CYCLES = env_params["max_cycles"]
        self.ACTION_SPACE = 6
        address = ADDRESS['ip'] + ":" + str(ADDRESS['port'])
        self.obs = startXsim(address).getObs()
        self.filt_my_plane = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 2 or x["ID"] == 11]
        self.AGENT_NAMES = self.returnAgent()
        self.N_AGENTS = 2
        self.OBS_SHAPE = (2, 20)
        self.agents_id = [2, 11]
        self.envs = None  # TODO 待处理
        self.observation_spaces = dict(
            zip(self.agents_id, gym.spaces.Box([-150000, 150000], [-150000, 150000], [0, 20000], (3,), np.float32)))

    def returnAgent(self):
        filt_my_plane = [x for x in self.obs["red"]["platforminfos"] if x["ID"] == 2 or x["ID"] == 11]
        return filt_my_plane  # TODO 红方的两架飞机, 未来需要修改

    def XsimEnvinit(self, XSimEnv):
        XSimEnv.__init__(self)
        self.agents = {}
        self.__init_agents()

        self.launch_missile = []
        self.last_red_entities = []
        self.last_blue_entities = []
        self.damage_entities = []

    def __init_agents(self):
        self.red_cls = config["agents"]['red']
        self.blue_cls = config["agents"]['blue']
        red_agent = self.red_cls('red', {"side": 'red'})
        self.agents["red"] = red_agent
        blue_agent = self.blue_cls('blue', {"side": 'blue'})
        self.agents["blue"] = blue_agent

    def __reset_agents(self):
        self.agents["red"].reset()
        self.agents["blue"].reset()

    def _reset(self):
        # 智能体重置
        # self.__init_agents()
        self.__reset_agents()
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
        for side, agent in self.agents.items():
            cmd_list = self._agent_step(agent, cur_time, obs[side], global_obs)
            # print(cmd_list)
            actions.extend(cmd_list)

        return actions

    def _agent_step(self, agent, cur_time, obs_side, global_obs):
        cmd_list = agent.step(cur_time, obs_side, global_obs)
        return cmd_list

    def get_done(self, obs):
        global red_area_score
        global blue_area_score

        # print(obs)
        done = [0, 0, 0]  # 终止标识， 红方战胜利， 蓝方胜利

        # 时间超时，终止
        cur_time = obs["sim_time"]
        # print("get_done cur_time:", cur_time)
        if cur_time >= 20 * 60 - 1:
            done[0] = 1
            red_round_score, blue_round_score = self._cal_score(obs)
            done = self._print_score(red_round_score, blue_round_score, "1", red_area_score, blue_area_score)

            # 重置区域得分
            red_area_score = 0
            blue_area_score = 0
            return done

        # 红方有人机全部战损就终止
        red_obs_units = obs["red"]["platforminfos"]
        filt_uav_plane = [x for x in obs["red"]["platforminfos"] if x["ID"] == 2 or x["ID"] == 11]
        no_red_leader = True
        if len(filt_uav_plane) == 2:
            no_red_leader = False
        for red_obs_unit in filt_uav_plane:
            if red_obs_unit["Type"] == 2:
                # 判断红方有人机是否在中心区域
                distance_to_center = math.sqrt(red_obs_unit["X"] * red_obs_unit["X"] +
                                               red_obs_unit["Y"] * red_obs_unit["Y"] +
                                               (red_obs_unit["Alt"] - 9000) * (red_obs_unit["Alt"] - 9000))
                # if distance_to_center <= CENTER_AREA_RADIO and red_obs_unit["Alt"] >= 2000 and red_obs_unit['Alt'] <= 16000:
                #     red_area_score = red_area_score + 1

        # 蓝方有人机全部战损就终止
        blue_obs_units = obs["blue"]["platforminfos"]
        no_blue_leader = True
        for blue_obs_unit in blue_obs_units:
            if blue_obs_unit["Type"] == 1:
                no_blue_leader = False
                # 判断蓝方有人机是否在中心区域
                distance_to_center = math.sqrt(blue_obs_unit["X"] * blue_obs_unit["X"] +
                                               blue_obs_unit["Y"] * blue_obs_unit["Y"] +
                                               (blue_obs_unit["Alt"] - 9000) * (blue_obs_unit["Alt"] - 9000))
                # if distance_to_center <= 0 and
                #         blue_obs_unit["Alt"] >= 2000 and
                #         blue_obs_unit['Alt'] <= 16000:
                #     blue_area_score = blue_area_score + 1
        if no_red_leader or no_blue_leader:
            red_round_score, blue_round_score = self._cal_score(obs)
            if no_red_leader and (not no_blue_leader):
                done_reason_code = "2"
            elif (not no_red_leader) and no_blue_leader:
                done_reason_code = "3"
            else:
                done_reason_code = "4"
            done = self._print_score(red_round_score, blue_round_score, done_reason_code)

            red_area_score = 0
            blue_area_score = 0
            return done

        return done

    @staticmethod
    def _cal_score(obs):

        # 计算剩余兵力
        red_leader = 0
        red_uav = 0
        red_missile = 0
        blue_leader = 0
        blue_uav = 0
        blue_missile = 0
        for unit in obs["red"]["platforminfos"]:
            red_missile += unit["LeftWeapon"]
            if unit["Type"] == 1:
                red_leader += 1
            else:
                red_uav += 1
        for unit in obs["blue"]["platforminfos"]:
            blue_missile += unit["LeftWeapon"]
            if unit["Type"] == 1:
                blue_leader += 1
            else:
                blue_uav += 1

        # # 计算剩余兵力与剩余导弹数的权重和
        #
        # red_round_score = red_leader * LEADER_SCORE_WEIGHT + \
        #                   red_uav * UAV_SCORE_WEIGHT + \
        #                   red_missile * MISSILE_SCORE_WEIGHT
        # blue_round_score = blue_leader * LEADER_SCORE_WEIGHT + \
        #                    blue_uav * UAV_SCORE_WEIGHT + \
        #                    blue_missile * MISSILE_SCORE_WEIGHT

        # return red_round_score, blue_round_score

    def print_logs(self, obs, num):
        global count

        filename = "logs/" + str(self.red_cls).split("'")[1] + "_VS_" + str(self.blue_cls).split("'")[1] + "_" + str(
            num) + ".txt"

        if num != count:
            if not os.path.isdir("logs"):
                os.mkdir("logs")
            if os.path.isfile(filename):
                os.remove(filename)
            count = num
            with open(filename, "w") as file:
                file.write("红方:" + str(self.red_cls).split("'")[1] + "\n")
                file.write("蓝方:" + str(self.blue_cls).split("'")[1] + "\n")
                file.close()

            self.last_red_entities = obs["red"]["platforminfos"]
            self.last_blue_entities = obs["blue"]["platforminfos"]
            self.launch_missile = []
            self.damage_entities = []

        with open(filename, "a") as fileobject:
            cur_time = obs["sim_time"]
            cur_red_entity_ids = []
            for cur_red_entity in obs["red"]["platforminfos"]:
                cur_red_entity_ids.append(cur_red_entity["ID"])
            for last_red_entity in self.last_red_entities:
                # 如果上一步长的实体不在当前步长实体内,说明该实体毁伤
                # 记录毁伤实体信息,用于匹配导弹信息
                if last_red_entity["ID"] not in cur_red_entity_ids:
                    self.damage_entities.append(last_red_entity)
                    entity_file = "[" + str(cur_time) + "][" + str(last_red_entity["Identification"]) + "]:[" + \
                                  last_red_entity[
                                      "Name"] + "]战损"
                    fileobject.writelines(entity_file + "\n")
            self.last_red_entities = obs["red"]["platforminfos"]

            cur_blue_entity_ids = []
            for cur_blue_entity in obs["blue"]["platforminfos"]:
                cur_blue_entity_ids.append(cur_blue_entity["ID"])
            for last_blue_entity in self.last_blue_entities:
                # 如果上一步长的实体不在当前步长实体内,说明该实体毁伤
                # 记录毁伤实体信息,用于匹配导弹信息
                if last_blue_entity["ID"] not in cur_blue_entity_ids:
                    self.damage_entities.append(last_blue_entity)
                    entity_file = "[" + str(cur_time) + "][" + str(last_blue_entity["Identification"]) + "]:[" + \
                                  last_blue_entity["Name"] + "]战损"
                    fileobject.writelines(entity_file + "\n")
            self.last_blue_entities = obs["blue"]["platforminfos"]

            all_entities = self.last_red_entities + self.last_blue_entities + self.damage_entities
            for missile in obs["red"]["missileinfos"]:
                if missile["ID"] not in self.launch_missile:
                    LauncherName = None
                    EngageTargetName = None
                    self.launch_missile.append(missile["ID"])
                    for entity in all_entities:
                        if entity["ID"] == missile["LauncherID"]:
                            LauncherName = entity["Name"]
                            break
                    for entity in all_entities:
                        if entity["ID"] == missile["EngageTargetID"]:
                            EngageTargetName = entity["Name"]
                            break
                    rocket_file = "[" + str(cur_time - 1) + "][" + str(
                        missile["Identification"]) + "]:[" + LauncherName + "]发射一枚导弹打击[" + EngageTargetName + "]"
                    fileobject.writelines(rocket_file + "\n")
