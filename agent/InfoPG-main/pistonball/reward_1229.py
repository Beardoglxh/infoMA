from utils.utils_math import TSVector3
import math
import numpy as np

"""奖励相关"""
HUMAN_REWARD = 500  # 有人机打击奖励
DRONE_REWARD = 10000  # 无人机打击奖励
HIT_BONUS = 50  # 打弹奖励
DISTANCE_COEFFICIENT = 10e6  # 距离系数比例(视情况调整)(1000m-150000m取完倒数后再乘以这个系数, 范围大致是6~1000)
LOG_COEFFICIENT = 1  # 对数比例系数(视情况调整)
VELOCITY_COEFFICIENT = 1  # 速度朝向系数比例(视情况调整)
HIT_PENALTY = -10000  # 被击中惩罚
OUT_PENALTY = -50  # 边界惩罚

"""参数相关"""
MISSILE_PLANE_DISTANCE = 1000  # 极限距离, 导弹距离飞机该范围以内则视为命中
DRONE_BULLET_COUNT = 2  # 无人机载弹量
MAX_TIME = 1200


class Bait_Reward:
    def __init__(self, my_planes, bait_planes, enemy_planes):
        self.my_planes = my_planes
        self.bait_planes = bait_planes
        self.enemy_planes = enemy_planes
        self.my_id_list = [x.ID for x in self.my_planes]
        self.bait_id_list = [x.ID for x in self.bait_planes]
        self.enemy_id_list = [x.ID for x in self.enemy_planes]
        self.reward_dict = {x.ID: 0 for x in self.bait_planes}

    def __call__(self, my_planes, bait_planes, enemy_planes):
        self.reset()
        bait_planes = [x for x in bait_planes]
        filt_planes = [x for x in my_planes]
        enemy_planes = [x for x in enemy_planes]
        my_planes = []
        my_planes.append(bait_planes)
        my_planes.append(filt_planes)
        self.interference(bait_planes, filt_planes, enemy_planes)
        self.distance_reward(bait_planes, filt_planes)
        # self.areareward(bait_planes, filt_planes, enemy_planes)
        return self.reward_dict

    def angle_reward(self, bait_planes):
        for bait_plane in bait_planes:
            if (bait_plane.Heading > 1) and (bait_plane.Heading < 2.14):
                self.reward_dict[bait_plane.ID] += 1

    def areareward(self, bait_planes, filt_planes, enemy_planes):
        for bait_plane in bait_planes:
            if bait_plane.X > filt_planes[0].X and bait_plane.X < enemy_planes[0].X and bait_plane.X < enemy_planes[1].X and bait_plane.X < enemy_planes[2].X:
                self.reward_dict[bait_plane.ID] += 1

    def distance_reward(self, bait_planes, filt_planes):
        for bait_plane in bait_planes:
            for filt_plane in filt_planes:
                distance = self.get_dis(bait_plane.pos3d, filt_plane.pos3d)
                if distance < 25000:
                    self.reward_dict[bait_plane.ID] += 1


    def interference(self, bait_my_planes, filt_my_planes, enemy_planes):
        # 诱饵机抗干扰
        filt_my_planes = [x for x in filt_my_planes]
        bait_my_planes = [x for x in bait_my_planes]
        pca_enemy_planes = [x for x in enemy_planes if (x.ID == 6 or x.ID == 19)]
        count = len(filt_my_planes)
        for filt_my_plane in filt_my_planes:
            # False不被干扰
            in_scope = False
            for pca_enemy_plane in pca_enemy_planes:
                # 在pca角度外或距离外continue
                # print("distance between {} and {}:".format(pca_enemy_plane["Name"], filt_my_plane["Name"]) + str(self.plane_distance(filt_my_plane, pca_enemy_plane)))
                if self.angle_lines(pca_enemy_plane, filt_my_plane) > 60 or self.get_dis(filt_my_plane.pos3d,
                                                                                                pca_enemy_plane.pos3d) > 80000:
                    continue
                # 在pca角度内并且小于30km，break，多功能被干扰
                if self.get_dis(filt_my_plane.pos3d, pca_enemy_plane.pos3d) < 30000:
                    count -= 1
                    # print(str(filt_my_plane['ID']) + "is in scape!_30000")
                    break
                # 在角度内距离在30-80km查看诱饵机
                in_bait_scope = False
                for bait_my_plane in bait_my_planes:
                    # 在诱饵机保护范围内则受保护
                    if self.angle_anti_interference(pca_enemy_plane, bait_my_plane, filt_my_plane) < 2.5:
                        self.reward_dict[bait_my_plane.ID] += 50
                        in_bait_scope = True
                        break
                if not in_bait_scope:
                    count -= 1
                    # print(str(filt_my_plane["Name"]) + "is in scape!")
                    break
        reward = count * 10

    def angle_lines(self, plane_a, plane_c):
        # 计算飞行器A/B/C的航线旋转矩阵

        # 计算飞行器A/B的位置向量, 并转换为numpy数组
        vector_A = np.array([plane_a.X, plane_a.Y])
        vector_C = np.array([plane_c.X, plane_c.Y])

        # 计算射线A和AC的方向向量
        direction_A = np.array([math.cos(plane_a.Heading - math.pi / 2), math.sin(plane_a.Heading - math.pi / 2)])  # 假设以A为起点，以A的航向为方向的射线与x轴平行
        direction_AC = vector_C - vector_A

        # 计算射线A和AC的夹角
        angle_A_AC = self.angle(direction_A, direction_AC)
        # print("{}:{} degree".format(plane_a["Name"], plane_c["Name"]) + str(math.degrees(angle_A_AC)))
        return math.degrees(angle_A_AC)

    def angle_anti_interference(self, plane_a, plane_b, plane_c):
        # # 计算飞行器A/B/C的航线旋转矩阵
        # matrix_A = self.euler_to_matrix(plane_a["Pitch"], plane_a["Heading"])
        # matrix_B = self.euler_to_matrix(plane_b["Pitch"], plane_b["Heading"])
        # matrix_C = self.euler_to_matrix(plane_c["Pitch"], plane_c["Heading"])
        # 计算飞行器A/B/C的位置向量，并转换为numpy数组
        vector_A = np.array([plane_a.X, plane_a.Y])
        vector_B = np.array([plane_b.X, plane_b.Y])
        vector_C = np.array([plane_c.X, plane_c.Y])

        # 计算射线AB和BC的方向向量
        direction_AB = vector_B - vector_A
        direction_BC = vector_C - vector_B

        # 计算射线AB和BC的夹角
        angle_AB_BC = self.angle(direction_AB, direction_BC)
        # print("{}:{}:{} degree".format(plane_a["Name"], plane_b["Name"], plane_c["Name"]) + str(math.degrees(angle_AB_BC)))
        return math.degrees(angle_AB_BC)

    def angle(self, v1, v2):
        # 判断v1和v2是否是numpy数组类型，并且长度相等
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray) and len(v1) == len(v2):
            # 计算v1和v2的点积和模长
            product = np.dot(v1, v2)
            length_v1 = np.linalg.norm(v1)
            length_v2 = np.linalg.norm(v2)
            # 判断分母是否为0，避免除零错误
            if length_v1 * length_v2 != 0:
                # 计算夹角的余弦值，并限制在[-1, 1]范围内，避免反余弦函数出错
                cos_value = product / (length_v1 * length_v2)
                cos_value = max(-1, min(cos_value, 1))
                # 计算夹角的弧度值，并返回
                radian = math.acos(cos_value)
                return radian
            else:
                # 如果分母为0，抛出异常
                raise ValueError("v1 and v2 must not be zero vectors")
        else:
            # 如果v1和v2不满足条件，抛出异常
            raise ValueError("v1 and v2 must be numpy arrays of equal length")

    def result_based_reward(self, obs, my_planes):
        self.reset()
        for plane in my_planes:
            self.reward_dict[plane.ID] += self.time_reward(obs)
            self.reward_dict[plane.ID] += self.missile_reward(plane)
        return self.reward_dict

    def missile_reward(self, my_plane):
        return my_plane.LeftWeapon * 50

    def time_reward(self, obs):
        return MAX_TIME - obs["sim_time"]

    def reset(self):
        self.reward_dict = {x.ID: 0 for x in self.bait_planes}

    # def distance_reward(self, my_planes, enemy_planes):
    #     for my_plane in my_planes:
    #         for enemy_plane in enemy_planes:
    #             distance = self.get_dis(my_plane.pos3d, enemy_plane.pos3d)
    #             reward = math.log(1 / distance * DISTANCE_COEFFICIENT) * LOG_COEFFICIENT
    #             self.reward_dict[my_plane.ID] += reward

    def velocity_reward(self, my_planes, target_plane):  # 速度奖励的话, 要不就设置和无人机的朝向去比较吧
        if not target_plane:
            return
        for my_plane in my_planes:
            target_x = target_plane.X - my_plane.X
            target_y = target_plane.Y - my_plane.Y
            tip = 1 if target_x > 0 else -1
            target_angle = tip * math.acos(target_y / math.sqrt(target_x ** 2 + target_y ** 2))
            theta = abs(my_plane.Heading - target_angle)
            if theta > math.pi:
                theta = 2 * math.pi - theta
            reward = my_plane.Speed * math.cos(theta) * VELOCITY_COEFFICIENT
            self.reward_dict[my_plane.ID] += reward

    def bullet_reward(self, my_planes):
        for my_plane in my_planes:
            reward = HIT_BONUS * (DRONE_BULLET_COUNT - my_plane.LeftWeapon)
            self.reward_dict[my_plane.ID] += reward

    def hit_reward(self, red_missiles, enemy_planes):
        if not red_missiles:
            return
        for missile in red_missiles:
            for enemy_plane in enemy_planes:
                if enemy_plane.ID == missile["EngageTargetID"]:
                    if self.get_dis(enemy_plane.pos3d, self.missile_xyz_to_pos3d(
                            missile)) < MISSILE_PLANE_DISTANCE:  # 打击奖励, 如果导弹和目标实体距离小于极限距离, 则对发射的实体编号进行加分.
                        if enemy_plane.ID == 26:
                            self.reward_dict[missile["LauncherID"]] += DRONE_REWARD  # 击中无人机
                        else:
                            self.reward_dict[missile["LauncherID"]] += HUMAN_REWARD  # 击中有人机

    def punishment_point(self, blue_missiles, my_planes):
        if not blue_missiles:
            return
        for missile in blue_missiles:
            for my_plane in my_planes:
                if my_plane.ID == missile["EngageTargetID"]:
                    if self.get_dis(my_plane.pos3d, self.missile_xyz_to_pos3d(
                            missile)) < MISSILE_PLANE_DISTANCE:  # 惩罚奖励, 如果蓝方导弹和目标实体小于极限距离, 则对目标实体距离进行扣分.
                        self.reward_dict[missile["EngageTargetID"]] += HIT_PENALTY

    def get_dis(self, pos1, pos2):
        return TSVector3.distance(pos1, pos2)

    def missile_xyz_to_pos3d(self, missile):
        pos3d = {"X": missile["X"], "Y": missile["Y"], "Z": missile["Alt"]}
        return pos3d


