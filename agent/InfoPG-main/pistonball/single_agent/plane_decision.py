import math
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3


class Decision:
    def __init__(self):
        self.cmd_list = list()
        self.hit_enemy_list = []
        self.min_value = -145000
        self.max_value = 145000

    def updateplaneinfo(self, my_plane, bait_my_plane, enemy_plane):
        self.my_plane = my_plane
        self.bait_my_plane = bait_my_plane
        self.enemy_plane = enemy_plane

    def switchcase(self, num: int, plane):
        self.cmd_list = []
        if num == 0:
            return self.fly(plane)
        elif num == 1:
            return self.turn_left(plane)
        elif num == 2:
            return self.turn_right(plane)
        elif num == 3:
            return self.up_height(plane)
        elif num == 4:
            return self.down_height(plane)

    def fly(self, plane):  # 飞机按原方向飞行
        cmd_list = []
        delta_x = plane.Speed * math.sin(plane.Heading) * 2
        delta_y = plane.Speed * math.cos(plane.Heading) * 2

        new_x = min(max(plane.X + delta_x, self.min_value), self.max_value)  # 保证数据不会超范围
        new_y = min(max(plane.Y + delta_y, self.min_value), self.max_value)
        new_z = plane.Z
        # cmd_list.append(env_cmd.make_linepatrolparam(plane.ID,
        #                                              [{"X": new_x, "Y": new_y, "Z": new_z}], plane.Speed, 1.0, 3))
        cmd_list.append(env_cmd.make_areapatrolparam(plane.ID, new_x, new_y, new_z, 2000, 2000, 300, 1.0, 4))
        return cmd_list

    def turn_left(self, plane):  # 左转的话, 就让heading小一点点就好了
        cmd_list = []
        change_x = 1
        change_y = 1
        delta_x = plane.Speed * math.sin(plane.Heading - 1.5)
        delta_y = plane.Speed * math.cos(plane.Heading - 1.5)
        if delta_x < 0:
            change_x = -1
        if delta_y < 0:
            change_y = -1
        new_x = min(max(plane.X + change_x * 2000, self.min_value), self.max_value)
        new_y = min(max(plane.Y + change_y * 2000, self.min_value), self.max_value)
        new_z = plane.Z
        # cmd_list.append(env_cmd.make_linepatrolparam(plane.ID,
        #                                                   [{"X": new_x, "Y": new_y, "Z": new_z}], plane.Speed, 1.0, 3))
        cmd_list.append(env_cmd.make_areapatrolparam(plane.ID, new_x, new_y, new_z, 2000, 2000, 300, 1.0, 4))
        return cmd_list

    def turn_right(self, plane):
        cmd_list = []
        change_x = 1
        change_y = 1
        delta_x = plane.Speed * math.sin(plane.Heading + 1.5)
        delta_y = plane.Speed * math.cos(plane.Heading + 1.5)
        if delta_x < 0:
            change_x = -1
        if delta_y < 0:
            change_y = -1
        new_x = min(max(plane.X + change_x * 2000, self.min_value), self.max_value)
        new_y = min(max(plane.Y + change_y * 2000, self.min_value), self.max_value)
        new_z = plane.Z
        # cmd_list.append(env_cmd.make_linepatrolparam(plane.ID,
        #                                              [{"X": new_x, "Y": new_y, "Z": new_z}], plane.Speed, 1.0, 3))
        cmd_list.append(env_cmd.make_areapatrolparam(plane.ID, new_x, new_y, new_z, 2000, 2000, 300, 1.0, 4))
        return cmd_list

    def up_height(self, plane):
        cmd_list = []
        new_x = plane.X
        new_y = plane.Y
        new_z = plane.Z + 500
        cmd_list.append(env_cmd.make_linepatrolparam(plane.ID,
                                                     [{"X": new_x, "Y": new_y, "Z": new_z}], 300, 1.0, 3))
        return cmd_list

    def down_height(self, plane):
        cmd_list = []
        new_x = plane.X
        new_y = plane.Y
        new_z = plane.Z - 500
        cmd_list.append(env_cmd.make_linepatrolparam(plane.ID,
                                                     [{"X": new_x, "Y": new_y, "Z": new_z}], 300, 1.0, 3))
        return cmd_list

    def attack(self, plane):
        threat_plane_list = self.get_threat_target_list()

        for threat_plane in threat_plane_list:
            attack_plane = self.can_attack_plane(threat_plane)

            if attack_plane is not None:
                if attack_plane.Type == 1:
                    self.cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 0.8))
                else:
                    self.cmd_list.append(env_cmd.make_attackparam(attack_plane.ID, threat_plane.ID, 1))
                # self.hit_enemy_list.append([threat_plane, None])
                threat_plane.num_locked_missile += 1

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

    def get_threat_target_list(self):
        # 有人机最重要，距离，带弹数量
        threat_dict = {}
        for enemy in self.enemy_plane:
            dis = 99999999
            for my_plane in self.my_plane:
                dis_tmp = TSVector3.distance(my_plane.pos3d, enemy.pos3d)
                if dis_tmp < dis:
                    dis = dis_tmp
            if enemy.Type == 1:
                # 敌机在距离我方有人机在距离的前提下会多20000的威胁值，并且敌人是有人机会再多10000威胁值
                dis -= 10000
            dis -= 20000

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
