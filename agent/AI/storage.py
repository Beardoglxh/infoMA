import numpy as np
import torch.nn as nn
import torch


class Batch_Storage:
    def __init__(self, buff_size, split_size, workers, agents_num):
        self.current_size = 0
        self.size = buff_size
        self.batch_size = 256
        self.workers = workers
        self.batch_split_size = split_size
        self.batch_storage = [[[] for _ in range(0, self.size)] for i in range(agents_num)]
        # self.batch_split_indeces = np.arange(0, self.batch_split_size).astype(np.uint8)
        self.agents_num = agents_num
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device

    def add_to_batch(self, observations, next_observations, initial, actions, rewards, dones=False):
        idxs = self._get_storage_idx(1)
        for agent_ix in range(0, self.agents_num):
            for id in idxs:
                self.batch_storage[agent_ix][id] = [observations[agent_ix], initial[agent_ix], rewards,
                                                    actions[agent_ix], next_observations[agent_ix], dones]

    def sample(self, batch_size):
        nums = []
        while True:
            num = np.random.randint(0, self.current_size)
            if self.batch_storage[0][num][4].shape ==(1, 10) and self.batch_storage[1][num][4].shape ==(1, 10) and num not in nums:
                nums.append(num)
            if len(nums) == 256:
                break

        idx = np.random.randint(0, self.current_size, batch_size)
        temp_buffer = {'o': [],
                       'initial': [],
                       'rewards': [],
                       'action': [],
                       'o_next': [],
                       'dones': []}
        for agent_ix in range(self.agents_num):
            temp_buffer['o'].append(
                torch.cat([torch.tensor(self.batch_storage[agent_ix][id][0], dtype=torch.float32) for id in nums],
                          dim=0))

            temp_buffer['initial'].append(torch.cat([self.batch_storage[agent_ix][id][1] for id in nums], dim=0))
            temp_buffer['rewards'].append(
                torch.cat([torch.tensor([self.batch_storage[agent_ix][id][2]]).unsqueeze(0) for id in nums], dim=0))
            temp_buffer['action'].append(
                torch.cat([torch.from_numpy(self.batch_storage[agent_ix][id][3]).unsqueeze(0) for id in nums], dim=0))
            temp_buffer['o_next'].append(
                torch.cat([torch.tensor(self.batch_storage[agent_ix][id][4].to(self.device), dtype=torch.float32) for id in nums],
                          dim=0))
            temp_buffer['dones'].append(
                torch.tensor([self.batch_storage[agent_ix][id][5] for id in nums], dtype=torch.bool).unsqueeze(1))
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        # inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        # if inc == 1:
        #     idx = idx[0]
        return idx
