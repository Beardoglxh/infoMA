import abc
from typing import List


class Agent(metaclass=abc.ABCMeta):
    def __init__(self, name, side, **kwargs):
        """必要的初始化"""
        self.name = name
        self.side = side

    @abc.abstractmethod
    def _init(self):
        pass

    def step(self, **kwargs) -> List[dict]:
        """输入态势信息，返回指令列表"""
        raise NotImplementedError

