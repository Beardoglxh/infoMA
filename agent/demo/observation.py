from agent.demo.agent_base import MyPlane, EnemyPlane, Missile

# 全局态势信息
class GlobalObservation(object):
    def __init__(self):
        # 全态势信息
        self.observation = Observation(MyPlane)
        # 感知态势信息
        self.perception_observation = Observation(EnemyPlane)
        # 全局导弹信息
        self.missile_observation = MissileObservation()

    # 更新态势
    def update_observation(self, obs):
        self.observation.update_observation(obs['platforminfos'])
        self.perception_observation.update_observation(obs['trackinfos'])
        self.missile_observation.update_observation(obs['missileinfos'])


# 态势信息
class Observation(object):
    def __init__(self, cls):
        self.agent_info_list = []
        self.lost_agent_info = []
        self.cls = cls

    # 更新态势
    def update_observation(self, obs_list: list):
        # 更新真实智能体信息
        for obs_agent_info in obs_list:
            agent_in_list = False
            for agent_info in self.agent_info_list:
                if obs_agent_info['ID'] == agent_info.ID:
                    agent_info.update_agent_info(obs_agent_info)
                    agent_in_list = True
                    break
            if not agent_in_list:
                self.agent_info_list.append(self.cls(obs_agent_info))

        # 更改死亡的智能体状态
        self.lost_agent_info.clear()
        temp_agent_list = self.agent_info_list.copy()
        for agent_info in temp_agent_list:
            agent_in_list = False
            for obs_agent_info in obs_list:
                if obs_agent_info['ID'] == agent_info.ID:
                    agent_in_list = True
                    break
            if not agent_in_list:
                self.agent_info_list.remove(agent_info)
                self.lost_agent_info.append(agent_info)

    # 通过多种类型获取实体信息
    def get_agent_info_by_types(self, agent_type_list: list) -> list:
        agent_list = []
        for agent_info in self.agent_info_list:
            if agent_info.Type in agent_type_list:
                agent_list.append(agent_info)
        return agent_list

    # 通过一种类型获取实体信息
    def get_agent_info_by_type(self, agent_type: int) -> list:
        agent_list = []
        for agent_info in self.agent_info_list:
            if agent_info.Type == agent_type:
                agent_list.append(agent_info)
        return agent_list

    # 通过实体ID获取实体信息
    def get_agent_info_by_id(self, agent_id: int) -> object:
        for agent_info in self.agent_info_list:
            if agent_info.ID == agent_id:
                return agent_info
        return None

    # 获取所有飞机
    def get_all_agent_list(self) -> list:
        return self.agent_info_list

    # 获取所有实体ID
    def get_all_agent_id(self) -> list:
        agent_id_list = []
        for agent_info in self.agent_info_list:
            agent_id_list.append(agent_info.ID)
        return agent_id_list


# 导弹信息
class MissileObservation(object):
    def __init__(self):
        self.missile_info_list = []

    # 更新态势
    def update_observation(self, missile_obs_list: list):
        # 更新导弹信息
        current_step_update_missile_id = []
        for obs_missile_info in missile_obs_list:
            agent_in_list = False
            for missile_info in self.missile_info_list:
                if obs_missile_info['ID'] == missile_info.ID:
                    missile_info.update_missile_info(obs_missile_info)
                    agent_in_list = True
                    break
            if not agent_in_list:
                self.missile_info_list.append(Missile(obs_missile_info))
            current_step_update_missile_id.append(obs_missile_info['ID'])

        # 移除消失导弹
        need_remove_missile = []
        for missile_info in self.missile_info_list:
            if missile_info.ID not in current_step_update_missile_id:
                need_remove_missile.append(missile_info)

        for remove_missile in need_remove_missile:
            self.missile_info_list.remove(remove_missile)

    # 获取导弹信息列表
    def get_missile_list(self) -> list:
        return self.missile_info_list

    # 通过被攻击者ID获取导弹信息
    def get_missile_info_by_attacked_id(self, attacked_id):
        missile_info_list = []
        for missile_info in self.missile_info_list:
            if missile_info.ID == attacked_id:
                missile_info_list.append(missile_info)
        return missile_info_list

    # 根据导弹ID判断导弹是否存在
    def get_missile_info_by_rocket_id(self, rocket_id):
        is_find = False
        for missile_info in self.missile_info_list:
            if missile_info.ID == rocket_id:
                is_find = True
        return is_find