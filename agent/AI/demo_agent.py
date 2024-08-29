from typing import List
import pandas as pd
from agent.agent import Agent
from agent.AI.observation import GlobalObservation
from agent.AI.decision_making import DemoDecision
from agent.AI.storage import Batch_Storage

class DemoAgent(Agent):
    """
        自定义智能体
    """
    def __init__(self, name, config):
        super(DemoAgent, self).__init__(name, config['side'])
        self._init()
        self.train_times = 0

    def _init(self):
        # 全局态势
        self.global_observation = GlobalObservation()
        # 指挥决策
        self.commond_decision = DemoDecision(self.global_observation)
        self.obs = []
        self.initial_actions = []
        self.actions = []
        self.storage = Batch_Storage(int(1e6), 1, 1, self.commond_decision.config.n_agents)

    def reset(self):
        print("system reset")
        self.commond_decision.reset()
        self.obs = []
        self.initial_actions = []
        self.actions = []
        self.train_times += 1
        if self.train_times % 50 == 0 and self.train_times != 0:
            self.commond_decision.policy.save_policy(self.train_times)
        return

    def step(self, sim_time, obs, global_obs, **kwargs) -> List[dict]:
        cmd_list = []
        self.global_observation.update_observation(obs)
        observations, initial_actions, actions, rewards = self.commond_decision.update_decision(sim_time, cmd_list, global_obs)
        if self.obs != []:
            self.storage.add_to_batch(self.obs, observations, self.initial_actions, self.actions, rewards)
        self.obs = observations
        self.initial_actions = initial_actions
        self.actions = actions
        if self.storage.current_size > 256:
            self.commond_decision.trainer(self.storage.sample(256))
        return cmd_list

    def final_step(self, sim_time, obs):
        cmd_list = []
        self.global_observation.update_observation(obs)
        observations, rewards = self.commond_decision.final_info(sim_time)
        self.save_rewards()
        if self.obs != []:
            self.storage.add_to_batch(self.obs, observations, self.initial_actions, self.actions, rewards, True)
        # self.obs = observations
        # self.initial_actions = initial_actions
        # self.actions = actions
        # if self.storage.current_size > 256:
        #     self.commond_decision.trainer(self.storage.sample(256))
        return cmd_list


    def save_rewards(self):
        rewards = []
        sum_rewards = []
        for agent_storage in self.storage.batch_storage:
            returns = 0
            for i, data in enumerate(agent_storage):
                if data == []:
                    sum_rewards.append(returns)
                    rewards.append(returns / i)
                    break
                returns += data[2]

        results = rewards + sum_rewards
        df = pd.DataFrame([results])
        df.to_csv('data.csv', mode='a', header=False, index=False)
