from utils.utils_math import TSVector3


class Reward():
    def enemy_attacked_point(self, enemy_planes):
        if enemy_planes == []:
            return 0
        reward = 0
        for x in enemy_planes:
            if x.ID == 6 and self.enemy_exist_obs[0] != -1:
                if x.Availability == 1:
                    self.enemy_exist_obs[0] = 1
                elif self.enemy_exist_obs[0] == 1:
                    self.enemy_exist_obs[0] = -1
                    reward += 500
            elif x.ID == 19 and self.enemy_exist_obs[1] != -1:
                if x.Availability == 1:
                    self.enemy_exist_obs[1] = 1
                elif self.enemy_exist_obs[1] == 1:
                    self.enemy_exist_obs[1] = -1
                    reward += 500
            elif x.ID == 26 and self.enemy_exist_obs[2] != -1:
                if x.Availability == 1:
                    self.enemy_exist_obs[2] = 1
                elif self.enemy_exist_obs[2] == 1:
                    self.enemy_exist_obs[2] = -1
                    reward += 10000
        return reward

    def attack_point(self, my_planes):
        bullet_count = 0
        for x in my_planes:
            if x.ID == 2 or x.ID == 11:
                bullet_count += x.LeftWeapon
        reward = (self.bullet_count - bullet_count) * 50
        self.bulle_count = bullet_count
        return reward

    def __init__(self, config, my_planes, enemy_planes):
        self.side = config.side
        self.position_init()
        self.state_init(my_planes, enemy_planes)

    def state_init(self, my_planes, enemy_planes):
        self.enemy_exist_obs = [0, 0, 0]
        state_value = 0
        planes = [x for x in my_planes if x.ID == 2 or x.ID == 11]
        bullet_count = 0
        for plane in planes:
            state_value -= self.get_dis(plane, self.aim) * 0.005 - 500
            bullet_count += plane.LeftWeapon
        self.bullet_count = bullet_count
        self.state = state_value
        self.value0 = state_value

    def position_init(self):
        self.aim = {"X": 140000 * self.side, "Y": 0, "Z": 9500}

    def get_dis(self, plane1, plane2):
        """计算两架飞机的距离

        Args:
            jet1: 1号战机的3d坐标，{"X": X, "Y": Y, "Z": Z}
            jet2: 2号战机的3d租坐标

        Returns:
            距离
        """
        pos1 = plane1.pos3d
        pos2 = plane2
        return TSVector3.distance(pos1, pos2)

    def __call__(self, my_planes, enemy_planes):
        state_value = 0
        planes = [x for x in my_planes if x.ID == 2 or x.ID == 11]
        for plane in planes:
            state_value -= self.get_dis(plane, self.aim) * 0.005 - 500
        reward = state_value - self.state + self.enemy_attacked_point(enemy_planes) + self.attack_point(my_planes)
        self.state = state_value
        if len(planes) < 2:
            reward -= 1000
        return reward

    def reset(self):
        self.state = self.value0
