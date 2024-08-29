from typing import List

from agent.agent import Agent
from agent.demo.observation import GlobalObservation
from agent.demo.decision_making import DemoDecision

class DemoAgent(Agent):
    """
        自定义智能体
    """
    def __init__(self, name, config):
        super(DemoAgent, self).__init__(name, config['side'])
        self._init()

    def _init(self):
        # 全局态势
        self.global_observation = GlobalObservation()
        # 指挥决策
        self.commond_decision = DemoDecision(self.global_observation)

    def reset(self):
        self.commond_decision.reset()

    def step(self, sim_time, obs, global_obs, **kwargs) -> List[dict]:

        cmd_list = []
        self.global_observation.update_observation(obs)
        self.commond_decision.update_decision(sim_time, cmd_list)
        return cmd_list