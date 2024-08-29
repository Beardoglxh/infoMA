import os
import math

from config import config
from env.xsim_env import XSimEnv


LEADER_SCORE_WEIGHT = 60  # 有人机得分权重
UAV_SCORE_WEIGHT = 5     # 无人机得分权重
MISSILE_SCORE_WEIGHT = 1  # 导弹得分权重
CENTER_AREA_RADIO = 50000

red_area_score = 0
blue_area_score = 0
count = 0

DONE_REASON_MSG = {
    "0": "未达到终止条件!",
    "1": "超时!",
    "2": "红方有人机全部毁伤!",
    "3": "蓝方有人机全部毁伤!",
    "4": "双方有人机全部毁伤!"
}


class EnvRunner(XSimEnv):
    def __init__(self, address):
        print("初始化 EnvRunner")
        XSimEnv.__init__(self, config['time_ratio'], address)
        self.agents = {}
        self.__init_agents()  # 在这里调用了demo_agent

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

    def system__reset_agents(self):
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
            if side == 1:
                continue
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
                if distance_to_center <= CENTER_AREA_RADIO and red_obs_unit["Alt"] >= 2000 and red_obs_unit['Alt'] <= 16000:
                    red_area_score = red_area_score + 1

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
                if distance_to_center <= CENTER_AREA_RADIO and \
                        blue_obs_unit["Alt"] >= 2000 and \
                        blue_obs_unit['Alt'] <= 16000:
                    blue_area_score = blue_area_score + 1
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

        # 计算剩余兵力与剩余导弹数的权重和

        red_round_score = red_leader * LEADER_SCORE_WEIGHT + \
                          red_uav * UAV_SCORE_WEIGHT + \
                          red_missile * MISSILE_SCORE_WEIGHT
        blue_round_score = blue_leader * LEADER_SCORE_WEIGHT + \
                           blue_uav * UAV_SCORE_WEIGHT + \
                           blue_missile * MISSILE_SCORE_WEIGHT

        return red_round_score, blue_round_score


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
                file.write("红方:"+ str(self.red_cls).split("'")[1] +"\n")
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
                    entity_file = "[" + str(cur_time) + "][" + str(last_red_entity["Identification"]) + "]:[" + last_red_entity[
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




    def _print_score(self, red_round_score, blue_round_score, done_reason_code = "0", red_area_score = 0, blue_area_score = 0):
        filename = "logs/" + str(self.red_cls).split("'")[1] + "_VS_" + str(self.blue_cls).split("'")[1] + "_" + str(
            count) + ".txt"
        done = [1, 0, 0]
        with open(filename, "a") as fileobject:
            # 输出结束原因
            fileobject.writelines("到达终止条件: " + DONE_REASON_MSG[done_reason_code] + "\n")
            # 胜负判断
            if red_round_score > blue_round_score:
                done[1] = 1
                fileobject.writelines("红方获胜!" + "\n")
            elif red_round_score < blue_round_score:
                done[2] = 1
                fileobject.writelines("蓝方获胜!" + "\n")
            else:
                if red_area_score or blue_area_score:
                    if red_area_score > blue_area_score:
                        done[1] = 1
                        fileobject.writelines("双方得分相同.但红方占据中心位置的时间较长!" + "\n")
                        fileobject.writelines("红方获胜!" + "\n")
                    elif red_area_score < blue_area_score:
                        done[2] = 1
                        fileobject.writelines("双方得分相同.但蓝方占据中心位置的时间较长!" + "\n")
                        fileobject.writelines("蓝方获胜!" + "\n")
                    else:
                        fileobject.writelines("双方得分相同.且双方占据中心位置的时间一样长!" + "\n")
                        fileobject.writelines("平局!" + "\n")
                else:
                    fileobject.writelines("平局!" + "\n")

            fileobject.writelines("红方得分:" + str(red_round_score) + "\n")
            fileobject.writelines("蓝方得分:" + str(blue_round_score) + "\n")

            return done



