import copy
import numpy as np

from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3


class DemoDecision():

    def __init__(self, global_observation):

        self.global_observation = global_observation

        self.init_info()

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
            if rocket.Identification == self.my_plane[0].Identification:
                my_missile.append(rocket)
            else:
                enemy_missile.append(rocket)
        self.enemy_missile = enemy_missile
        self.my_missile = my_missile

        # 编队并更新编队信息（待测试）
        # self.formation()

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

    def update_decision(self, sim_time, cmd_list):
        self.update_entity_info(sim_time)
        # self.init_move(cmd_list)

        # 更新敌人的被打击列表
        undetected_list = self.update_hit_list()
        if self.side == 1:
            threat_plane_list = self.get_threat_target_list()
            if sim_time % 50 == 0:
                for threat_plane in threat_plane_list:
                    threat_plane.num_locked_missile = 0
            for threat_plane in threat_plane_list:
                attack_plane = self.can_attack_plane(threat_plane)

                if attack_plane is not None:
                    if attack_plane.Type == 1 and threat_plane.num_locked_missile == 0:
                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 0.8))
                    elif attack_plane.Type == 2 and threat_plane.num_locked_missile == 0:
                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 1))
                    self.hit_enemy_list.append([threat_plane, None])
                    threat_plane.num_locked_missile += 1

        if self.side == -1:
            # 开火模块
            threat_plane_list = self.get_threat_target_list()
            for threat_plane in threat_plane_list:
                attack_plane = self.can_attack_plane(threat_plane)

                if attack_plane is not None:
                    if attack_plane.Type == 1 and threat_plane.num_locked_missile == 0:
                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 0.8))
                    elif attack_plane.Type == 2 and threat_plane.num_locked_missile == 0:
                        cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 1))
                    self.hit_enemy_list.append([threat_plane, None])
                    threat_plane.num_locked_missile += 1

        self.update_evade()
        if self.side == -1:
            # 躲避模块
            for comba in self.evade_list:
                plane = comba["plane_entity"]
                enemy = comba["missile_entity"]
                plane.evade(enemy, cmd_list)

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
                env_cmd.make_entityinitinfo(leader_plane_1.ID, -150000 * self.side, 75000, 9500, 200, init_direction))
            cmd_list.append(
                env_cmd.make_entityinitinfo(leader_plane_1.ID, -150000 * self.side, -75000, 9500, 200, init_direction))

            for key, value in self.first_leader_formation.items():
                if key == "up_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, 37500, 9500, 200, init_direction))
                if key == "down_plane":
                    cmd_list.append(
                        env_cmd.make_entityinitinfo(value.ID, -125000 * self.side, -37500, 9500, 200, init_direction))

            for key, value in self.sec_leader_formation.items():
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
            cmd_list.append(
                env_cmd.make_entityinitinfo(self.first_formation["uav_1"].ID, -140000 * self.side, 0, 9500, 200,
                                            init_direction))

    def init_move(self, cmd_list):
        if self.side == -1:
            cmd_list.append(
                env_cmd.make_areapatrolparam(6,
                                             -140000, -75000, 9500, 2000, 2000, 400, 1.0, 4.0))
            cmd_list.append(
                env_cmd.make_areapatrolparam(30,
                                             -140000, 75000, 9500, 2000, 2000,  400, 1.0, 4.0))
            cmd_list.append(
                env_cmd.make_areapatrolparam(33,
                                             -140000, 0, 9500, 2000, 2000, 300, 1.0, 4.0))
            cmd_list.append(
                env_cmd.make_areapatrolparam(31,
                                             -140000, -75000, 9500, 2000, 2000, 400, 1.0, 4.0))
            cmd_list.append(
                env_cmd.make_areapatrolparam(32,
                                             -140000, 75000, 9500, 2000, 2000,  400, 1.0, 4.0))
            cmd_list.append(
                env_cmd.make_areapatrolparam(34,
                                             -140000, 0, 9500, 2000, 2000, 300, 1.0, 4.0))


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
                # 保护策略, 如果有敌机飞过我方有人机则会优先打击.
                if enemy.X > my_plane.X:
                    dis = 0
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
        # for hit_enemy in self.hit_enemy_list:
        #     leader_hit = False
        #     # 敌有人机可以打两发
        #     if hit_enemy[0].num_locked_missile == 1 and hit_enemy[0].Type == 1:
        #         leader_hit = True
        #     for threat_plane in threat_plane_list:
        #         if hit_enemy[0] == threat_plane and not leader_hit:
        #             threat_plane_list.remove(threat_plane)
        return threat_plane_list

    def can_attack_plane(self, enemy_plane):
        attack_plane = None
        dis = 9999999

        my_plane = [x for x in self.my_plane if x.ID != 24 and x.ID != 25 and x.ID != 26
                    and x.ID != 27 and x.ID != 28 and x.ID != 29 and x.ID != 10 and x.ID != 11]
        for my_plane in my_plane:
            tmp_dis = TSVector3.distance(my_plane.pos3d, enemy_plane.pos3d)
            if my_plane.Type == 1:
                left_weapon = my_plane.LeftWeapon > 0
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

    def activate_jam(self, cmd_list):
        for comba in self.jam_list:
            if comba[1] > 10:
                comba[1] = 0
                cmd_list.append(env_cmd.make_jamparam(comba[0].ID))
            else:
                comba[1] += 1
