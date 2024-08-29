from utils.utils_math import TSVector3


class Reward:
    def __init__(self, my_planes, enemy_planes):
        self.side = 1
        self.position_init()
        self.state_init(my_planes, enemy_planes)

    def position_init(self):
        self.aim = {"X": 140000 * self.side, "Y": 0, "Z": 9500}

    def state_init(self, my_planes, enemy_planes):
        self.enemy_exist_obs = [1, 1, 1]
        state_value1 = 0
        state_value2 = 0
        planes = [x for x in my_planes if x.ID == 2 or x.ID == 11]
        for plane in planes:
            if plane.ID == 2:
                state_value1 -= self.get_dis(plane, self.aim) * 0.005 - 500
            elif plane.ID == 11:
                state_value2 -= self.get_dis(plane, self.aim) * 0.005 - 500
        self.state1 = state_value1
        self.state2 = state_value2
        self.vlaue1 = state_value1
        self.vlaue2 = state_value2

    def get_dis(self, plane1, plane2):
        pos1 = plane1.pos3d
        pos2 = plane2
        return TSVector3.distance(pos1, pos2)

    def enemy_attacked_point(self, enemy_planes, missile):
        reward = 0
        for x in enemy_planes:
            if missile.EngageTargetID == 6:
                if x.ID == 6 and self.enemy_exist_obs[0] != -1:
                    if x.Availability != 1:
                        self.enemy_exist_obs[0] = -1
                        reward += 1000
            elif missile.EngageTargetID == 19:
                if x.ID == 19 and self.enemy_exist_obs[1] != -1:
                    if x.Availability != 1:
                        self.enemy_exist_obs[1] = -1
                        reward += 1000
            elif missile.EngageTargetID == 26:
                if x.ID == 26 and self.enemy_exist_obs[2] != -1:
                    if x.Availability != 1:
                        self.enemy_exist_obs[2] = -1
                        reward += 10000
        return reward

    def __call__(self, my_planes, enemy_planes, my_missile):
        self.my_exist_obs = [1, 1]
        state_value1 = 0
        state_value2 = 0
        planes = [x for x in my_planes if x.ID == 2 or x.ID == 11]
        for plane in planes:
            if plane.ID == 2:
                state_value1 -= self.get_dis(plane, self.aim) * 0.005 - 500
            elif plane.ID == 11:
                state_value2 -= self.get_dis(plane, self.aim) * 0.005 - 500
        for missile in my_missile:
            if missile.LauncherID == 2:
                reward1 = state_value1 - self.state1 + self.enemy_attacked_point(enemy_planes, missile)
            elif missile.LauncherID == 11:
                reward2 = state_value2 - self.state2 + self.enemy_attacked_point(enemy_planes, missile)
        self.state1 = state_value1
        self.state2 = state_value2
        for x in my_planes:
            if x.ID == 2 and self.my_exist_obs[0] != -1:
                if x.Availability != 1:
                    self.my_exist_obs[0] = -1
                    reward1 -= 10000
            elif x.ID == 11 and self.my_exist_obs[1] != -1:
                if x.Availability != 1:
                    self.my_exist_obs[1] = -1
                    reward2 -= 10000
        return reward1, reward2

    def reset(self):
        self.state1 = self.vlaue1
        self.state2 = self.vlaue2
