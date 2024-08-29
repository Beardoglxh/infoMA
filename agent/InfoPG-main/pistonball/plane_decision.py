import math
from env.env_cmd import CmdEnv as env_cmd
from utils.utils_math import TSVector3


class Decision:
    def __init__(self):
        self.cmd_list = list()
        self.hit_enemy_list = []
        self.min_value = -145000
        self.max_value = 145000

    def updateplaneinfo(self,DY_1, YE_2, enemy):
        self.my_plane = DY_1
        self.my_plane = YE_2
        self.enemy_plane = enemy

    def switchcase(self, num: int, plane):
        self.cmd_list = []
        if num == 0:
            return self.fly(plane)
        elif num == 1:
            return self.turn_left(plane)
        elif num == 2:
            return self.turn_right(plane)
        elif num == 3:
            return self.adjust_height(plane)
        elif num == 4:
            return self.down_height(plane)

    def fly(self, plane):  # 飞机按原方向飞行
        cmd_list = []
        delta_x = plane.Speed * math.sin(plane.Heading) * 2
        delta_y = plane.Speed * math.cos(plane.Heading) * 2

        new_x = min(max(plane.X + delta_x, self.min_value), self.max_value)  # 保证数据不会超范围
        new_y = min(max(plane.Y + delta_y, self.min_value), self.max_value)
        new_z = plane.Z
        if new_z > 10000:
            new_z = 10000
        if new_z < 2000:
            new_z =2000
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
        if new_z > 10000:
            new_z = 10000
        if new_z < 2000:
            new_z =2000
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
        if new_z > 10000:
            new_z = 10000
        if new_z < 2000:
            new_z =2000
        # cmd_list.append(env_cmd.make_linepatrolparam(plane.ID,
        #                                              [{"X": new_x, "Y": new_y, "Z": new_z}], plane.Speed, 1.0, 3))
        cmd_list.append(env_cmd.make_areapatrolparam(plane.ID, new_x, new_y, new_z, 2000, 2000, 300, 1.0, 4))
        return cmd_list

    def adjust_height(self, plane):
        cmd_list = []
        new_x = plane.X
        new_y = plane.Y
        new_z = min(plane.Z + 500, 15000)
        if new_z > 10000:
            new_z = 10000
        if new_z < 2000:
            new_z =2000
        cmd_list.append(env_cmd.make_linepatrolparam(plane.ID,
                                                     [{"X": new_x, "Y": new_y, "Z": new_z}], 300, 1.0, 3))
        return cmd_list

    def down_height(self, plane):
        cmd_list = []
        new_x = plane.X
        new_y = plane.Y
        # new_z = plane.Z - 500
        new_z = min(plane.Z - 500, 2000)
        if new_z > 10000:
            new_z = 10000
        if new_z < 2000:
            new_z =2000
        cmd_list.append(env_cmd.make_linepatrolparam(plane.ID,
                                                     [{"X": new_x, "Y": new_y, "Z": new_z}], 300, 1.0, 3))
        return cmd_list
