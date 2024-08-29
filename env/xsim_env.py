from config import IMAGE
from env.xsim_manager import XSimManager
from env.communication_service import CommunicationService



class XSimEnv(object):
    def __init__(self, time_ratio: int, address: str):
        # xsim引擎控制器
        self.xsim_manager = XSimManager(time_ratio, address, IMAGE)
        # 与xsim引擎交互通信服务
        self.communication_service = CommunicationService(self.xsim_manager.address)

    # def __del__(self):
    #     self.xsim_manager.close_env()

    def step(self, action: list) -> dict:
        try:
            obs = self.communication_service.step(action)
            return obs
        except Exception as e:
            print(e)
        # return self.communication_service.step(action)

    def reset(self):
        return self.communication_service.reset()

    def end(self):
        return self.communication_service.end()

    def close(self) -> bool:
        self.communication_service.close()
        # self.xsim_manager.close_env()
        return True
