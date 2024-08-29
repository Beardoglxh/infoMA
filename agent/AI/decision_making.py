import copy
import numpy as np
import pandas as pd
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3
from agent.AI.policy import Planes_Policy
import json
import torch
from agent.AI.reward import Reward


class config:
    def __init__(self):
        self.n_agents = 4
        self.input_size = 10
        self.save_dir = "./agent/AI/model"
        self.scenario_name = "4v3"
        self.high_action = 1
        self.action_shape = 6
        self.tau = float(0.01)
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.gamma = 0.95
        self.save_rate = 5000
        self.side = 1
        # self.plane_id = [2, 11, 12, 13]
        self.plane_id = [2, 11, 12, 13]
        # 有人机类型为1 无人机为0
        self.plane_types = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        self.noise_rate = float(0.1)
        self.epsione = float(0.1)
        self.plane_ability = {
            "leader": {
                "speed": [150, 400],
                "accelerate": 1,
                "height": [2000, 15000],
                "overload": 6
            },
            "uav": {
                "speed": [100, 300],
                "accelerate": 2,
                "height": [2000, 10000],
                "overload": 12
            }
        }


class DemoDecision():

    def __init__(self, global_observation):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.empty_cache()
            print('**Using:  for inference', torch.cuda.get_device_name(device))
        else:
            device = torch.device("cpu")
            print('**Using: cpu for inference')
        self.device = device

        # args = config()
        self.config = config()

        self.global_observation = global_observation
        self.init_info()
        self.policy = Planes_Policy(self.config, self.device, k_levels=1)

    def init_info(self):
        # 我方所有飞机信息列表
        self.my_plane = []
        # 我方有人机信息列表
        self.my_leader_plane = []
        # 我方无人机信息列表
        self.my_uav_plane = []

        # 敌方所有飞机信息列表
        self.enemy_plane = []
        # 敌方有人机信息列表
        self.enemy_leader_plane = []
        # 敌方无人机信息列表
        self.enemy_uav_plane = []

        # 我方导弹信息
        self.my_missile = []
        # 敌方导弹信息
        self.enemy_missile = []

        # 被打击的敌机列表
        self.hit_enemy_list = []

        # 识别红军还是蓝军
        self.side = None

        # 有人机第一编队
        self.first_leader_formation = {}
        # 有人机第二编队
        self.sec_leader_formation = {}

        # 无人机第一编队
        self.first_uav_formation = [0, 0]
        # 无人机第二编队
        self.sec_uav_formation = [0, 0]

        # 第一编队
        self.first_formation = {}
        # 第二编队
        self.sec_formation = {}

        # 躲避列表
        self.evade_list = []

    def reset(self):
        """当引擎重置会调用,选手需要重写此方法"""
        self.init_info()
        self.rewards.reset()

    def update_entity_info(self, sim_time):
        # 己方所有飞机信息
        self.my_plane = self.global_observation.observation.get_all_agent_list()
        # 己方有人机信息
        self.my_leader_plane = self.global_observation.observation.get_agent_info_by_type(1)
        # 己方无人机信息
        self.my_uav_plane = self.global_observation.observation.get_agent_info_by_type(2)
        # 敌方所有飞机信息
        self.enemy_plane = self.global_observation.perception_observation.get_all_agent_list()
        # 敌方有人机信息
        self.enemy_leader_plane = self.global_observation.perception_observation.get_agent_info_by_type(1)
        # 敌方无人机信息
        self.enemy_uav_plane = self.global_observation.perception_observation.get_agent_info_by_type(2)

        # 获取队伍标识
        if self.side is None:
            if self.my_plane[0].Identification == "红方":
                self.side = 1
            else:
                self.side = -1

        # 获取双方导弹信息
        missile = self.global_observation.missile_observation.get_missile_list()
        enemy_missile = []
        my_missile = []
        for rocket in missile:
            # 待测试
            if rocket.Identification == "红方":
                my_missile.append(rocket)
            else:
                enemy_missile.append(rocket)
        self.enemy_missile = enemy_missile
        self.my_missile = my_missile
        # 编队并更新编队信息（待测试）
        # self.formation()

    def update_decision(self, sim_time, cmd_list, global_obs):
        self.update_entity_info(sim_time)
        if sim_time <= 2:
            self.init_pos(sim_time, cmd_list)
            if sim_time == 1:
                self.rewards = Reward(self.config, self.my_plane, self.enemy_plane)
            return [], [], [], 0
        else:
            observations, initial, results = self.init_move(cmd_list, global_obs)
            reward = self.rewards(self.my_plane, self.enemy_plane)
            # 更新敌人的被打击列表
            # undetected_list = self.update_hit_list()

            # 开火模块
            threat_plane_list = self.get_threat_target_list()

            for threat_plane in threat_plane_list:
                attack_plane = self.can_attack_plane(threat_plane)

                if attack_plane is not None:
                    if attack_plane.Type == 1:
                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 0.8))
                    else:
                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 1))
                    self.hit_enemy_list.append([threat_plane, None])
                    threat_plane.num_locked_missile += 1
            #
            # # 制导
            # evade_plane_id = [plane.ID for plane in self.evade_list]
            # for enemy_plane in undetected_list:
            #     free_plane = []
            #     for my_plane in self.my_plane:
            #         if my_plane.ID not in evade_plane_id:
            #             free_plane.append(my_plane)
            #     dis = 999999
            #     guide_plane = None
            #     for my_plane in free_plane:
            #         tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane[0].pos3d)
            #         if tmp_dis < dis:
            #             guide_plane = my_plane
            #             dis = tmp_dis
            #
            #     if guide_plane is not None:
            #         cmd_list.append(
            #             env_cmd.make_areapatrolparam(guide_plane.ID, enemy_plane[1]["X"], enemy_plane[1]["Y"],
            #                                          enemy_plane[1]["Z"], 200, 100, 300, 1, 6))
            #
            # # 躲避模块
            # self.update_evade()
            # for comba in self.evade_list:
            #     plane = comba["plane_entity"]
            #     enemy = comba["missile_entity"]
            #     plane.evade(enemy, cmd_list)

            # self.activate_jam(cmd_list)

            return observations, initial, results, reward

    def trainer(self, transitions):
        self.policy.trainer(transitions)

    def formation(self):
        self.first_leader_formation["up_plane"] = self.my_uav_plane[0]
        self.first_leader_formation["down_plane"] = self.my_uav_plane[1]
        self.first_leader_formation["leader"] = self.my_leader_plane[0]
        self.sec_leader_formation["up_plane"] = self.my_uav_plane[2]
        self.sec_leader_formation["down_plane"] = self.my_uav_plane[3]
        self.sec_leader_formation["leader"] = self.my_leader_plane[1]

        self.first_uav_formation[0] = self.my_uav_plane[4]
        self.first_uav_formation[1] = self.my_uav_plane[5]

        self.sec_uav_formation[0] = self.my_uav_plane[6]
        self.sec_uav_formation[1] = self.my_uav_plane[7]

        self.first_formation["up_plane"] = self.my_uav_plane[0]
        self.first_formation["down_plane"] = self.my_uav_plane[1]
        self.first_formation["leader_plane"] = self.my_leader_plane[0]
        self.first_formation["uav_1"] = self.my_uav_plane[4]
        self.first_formation["uav_2"] = self.my_uav_plane[5]

        self.sec_formation["up_plane"] = self.my_uav_plane[2]
        self.sec_formation["down_plane"] = self.my_uav_plane[3]
        self.sec_formation["leader_plane"] = self.my_leader_plane[1]
        self.sec_formation["uav_1"] = self.my_uav_plane[6]
        self.sec_formation["uav_2"] = self.my_uav_plane[7]

        # 干扰
        self.jam_list = [[plane, 0] for plane in self.my_leader_plane]

    def init_pos(self, sim_time, cmd_list):
        self.formation()
        # 初始化部署
        if sim_time == 2:
            init_direction = 90
            if self.side == -1:
                init_direction = 270
            leader_plane_1 = self.first_leader_formation["leader"]
            leader_plane_2 = self.sec_leader_formation["leader"]
            # 初始化有人机位置
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_1.ID, -145000 * self.side, 75000, 9500, 200, init_direction))
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_1.ID, -145000 * self.side, -75000, 9500, 200, init_direction))

            for key, value in self.first_leader_formation.items():
                if key == "up_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, 85000, 9500, 200, init_direction))
                if key == "down_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, 65000, 9500, 200, init_direction))

            for key, value in self.first_leader_formation.items():
                if key == "up_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, -65000, 9500, 200, init_direction))
                if key == "down_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, -85000, 9500, 200, init_direction))

            for i, plane in enumerate(self.first_uav_formation):
                cmd_list.append(
                    env_cmd.make_entityinitinfo(plane.ID, -140000 * self.side, 65000 - ((i + 1) * 10000), 9500, 200,
                                                init_direction))

            for i, plane in enumerate(self.sec_uav_formation):
                cmd_list.append(
                    env_cmd.make_entityinitinfo(plane.ID, -140000 * self.side, -65000 + ((i + 1) * 10000), 9500, 200,
                                                init_direction))

    def init_move(self, cmd_list, global_obs):
        self.global_obs = global_obs
        filt_my_plane = [x for x in self.my_plane if self.side == 1 and (
                x.ID == 2 or x.ID == 11 or x.ID == 12 or x.ID == 13)]
        enemy_plane = [x for x in global_obs["blue"]["platforminfos"] if x["ID"] == 6]
        cmds, observations, initial, result = self.policy(filt_my_plane, enemy_plane, global_obs)
        cmd_list += cmds
        else_my_plane = [x for x in self.my_plane if x not in filt_my_plane]
        for i in else_my_plane:
            cmd_list.append(
                env_cmd.make_linepatrolparam(i.ID,
                                             [{"X": -150000 * self.side, "Y": i.Y, "Z": i.Z}], 250, 1.0, 3)
            )
        return observations, initial, result

    def _is_dead(self, plane):
        if plane.Availability == 0:
            return True
        else:
            return False

    def update_evade(self):
        missile_list = self.global_observation.missile_observation.get_missile_list()
        # missile_id = [missile.ID for missile in missile_list]
        evade_list = copy.deepcopy(self.evade_list)
        evade_id = [comb["missile_entity"].ID for comb in evade_list]

        # 统计所有被导弹瞄准的飞机
        if len(missile_list) != 0:
            for missile in missile_list:
                attacked_plane = self.global_observation.observation.get_agent_info_by_id(missile.EngageTargetID)
                if attacked_plane is None:
                    continue
                if missile.ID not in evade_id:
                    evade_comb = {}
                    evade_comb["plane_entity"] = attacked_plane
                    evade_comb["missile_entity"] = missile
                    evade_list.append(evade_comb)
            # 给危险程度分类 TODO

        # 将不需要躲避的移除列表
        evade_list = copy.deepcopy(self.evade_list)
        for attacked_plane in evade_list:
            missile = attacked_plane["missile_entity"]
            plane = attacked_plane["plane_entity"]
            missile_gone, over_target, safe_distance = False, False, False
            # 导弹已爆炸
            if not self.global_observation.missile_observation.get_missile_info_by_rocket_id(missile.ID):
                missile_gone = True
            # 过靶
            missile_vector_3d = TSVector3.calorientation(missile.Heading, missile.Pitch)
            missile_vector = np.array([missile_vector_3d["X"], missile_vector_3d["Y"]])
            missile_mp_vector_3d = TSVector3.minus(plane.pos3d, missile.pos3d)
            missile_mp_vector = np.array([missile_mp_vector_3d["X"], missile_mp_vector_3d["Y"]])
            res = np.dot(np.array(missile_vector), np.array(missile_mp_vector)) / (
                    np.sqrt(np.sum(np.square(np.array(missile_vector)))) + 1e-9) / (
                          np.sqrt(np.sum(np.square(np.array(missile_mp_vector)))) + 1e-9)
            if abs(res) > 1:
                res = res / abs(res)
            dir = np.math.acos(res) * 180 / np.math.pi
            if abs(dir) > 90:
                over_target = True
            # 飞到安全距离
            distance = TSVector3.distance(missile.pos3d, plane.pos3d)
            if distance >= 100000:
                safe_distance = True

            if any([missile_gone, over_target, safe_distance]):
                evade_list.remove(attacked_plane)

        self.evade_list = evade_list

    def update_attack(self):
        pass

    def get_threat_target_list(self):
        # 有人机最重要，距离，带弹数量
        threat_dict = {}
        for enemy in self.enemy_plane:
            dis = 99999999
            for my_plane in self.my_leader_plane:
                dis_tmp = TSVector3.distance(my_plane.pos3d, enemy.pos3d)
                if dis_tmp < dis:
                    dis = dis_tmp
            if enemy.Type == 1:
                # 敌机在距离我方有人机在距离的前提下会多20000的威胁值，并且敌人是有人机会再多10000威胁值
                dis -= 10000
            dis -= 20000

            for my_plane in self.my_uav_plane:
                dis_tmp = TSVector3.distance(my_plane.pos3d, enemy.pos3d)
                if dis_tmp < dis:
                    dis = dis_tmp

            if dis < 0:
                dis = 0
            if dis not in threat_dict:
                threat_dict[dis] = enemy
            else:
                threat_dict[dis + 0.1] = enemy

        threat_plane_list = [value for key, value in sorted(threat_dict.items(), key=lambda d: d[0])]
        for hit_enemy in self.hit_enemy_list:
            leader_hit = False
            # 敌有人机可以打两发
            if hit_enemy[0].num_locked_missile == 1 and hit_enemy[0].Type == 1:
                leader_hit = True
            for threat_plane in threat_plane_list:
                if hit_enemy[0] == threat_plane and not leader_hit:
                    threat_plane_list.remove(threat_plane)
        return threat_plane_list

    def can_attack_plane(self, enemy_plane):
        attack_plane = None
        dis = 9999999
        for my_plane in self.my_plane:
            tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
            if my_plane.Type == 1:
                left_weapon = my_plane.LeftWeapon > 1
            else:
                left_weapon = my_plane.LeftWeapon > 0
            in_range = my_plane.can_attack(tmp_dis)
            if (in_range and left_weapon) and tmp_dis < dis:
                dis = tmp_dis
                attack_plane = my_plane

        return attack_plane

    def update_hit_list(self):
        for comba in self.hit_enemy_list:
            if comba[1] is None:
                for my_missile in self.my_missile:
                    if my_missile.EngageTargetID == comba[0].ID:
                        comba[1] = my_missile
        undetected_list = []
        for comba in self.hit_enemy_list:
            is_dead = False
            if self.global_observation.perception_observation.get_agent_info_by_id(comba[0].ID):
                is_dead = self._is_dead(comba[0])
            else:
                undetected_list.append((comba[0], comba[0].pos3d))
            missile_gone = False
            if comba[1] is not None:
                if not self.global_observation.missile_observation.get_missile_info_by_rocket_id(comba[1].ID):
                    missile_gone = True
            if is_dead or missile_gone:
                self.hit_enemy_list.remove(comba)

        return undetected_list

    def final_info(self, sim_time):
        self.update_entity_info(sim_time)
        obervations = []
        enemy_plane = [x for x in self.global_obs["blue"]["platforminfos"] if x["ID"] == 6]
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

        if enemy_plane == []:
            enemy_info = torch.zeros(5).unsqueeze(dim=0)

        for plane_id in self.config.plane_id:
            obervation = []
            for plane in self.my_plane:
                if plane.ID == plane_id:
                    obervation.append(plane.Heading)
                    obervation.append(plane.Speed)
                    obervation.append(plane.X)
                    obervation.append(plane.Y)
                    obervation.append(plane.Z)
                    obervation = torch.tensor(obervation).unsqueeze(dim=0)
                    obervation = torch.cat((obervation, enemy_info), dim=1)
                    break
            if obervation == []:
                obervation = torch.zeros(5).unsqueeze(dim=0)
                obervation = torch.cat((obervation, enemy_info), dim=1)
            obervations.append(obervation)
        reward = self.rewards(self.my_plane, self.enemy_plane)
        return obervations, reward
