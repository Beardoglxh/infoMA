from time import sleep
from multiprocessing import Pool

from config import ADDRESS, config, ISHOST, XSIM_NUM
from env.env_runner import EnvRunner


# 启动单个XSIM
class BattleRunnerSignal(EnvRunner):
    def __init__(self, address):
        EnvRunner.__init__(self, address)  # 仿真环境初始化

    def run(self, num_episodes, map_start = "N"):
        battle_results = [0,0,0]  # [红方获胜局数, 蓝方获胜局数, 平局数量]
        for i in range(num_episodes):
            obs = self.step([])
            # print("===> init obs:", obs)
            done = [0,0,0]
            while True:
                if obs is None:
                    sleep(0.2)
                    obs = self.step([])
                    continue
                # print("obs: ", obs["sim_time"])
                if ISHOST:
                    self.print_logs(obs, i + 1)
                # done = self.get_done(obs)  # 推演结束(分出胜负或达到最大时长)
                # print("done:", done)
                if done[0]:  # 对战结束后环境重置
                    print("到达终止条件！！！！")
                    if done[1] > done[2]:
                        battle_results[0] += 1
                        print("第", i + 1, "局 ：  红方胜")
                    elif done[1] < done[2]:
                        battle_results[1] += 1
                        print("第", i + 1, "局：   蓝方胜")
                    else:
                        battle_results[2] += 1
                        print("第", i + 1, "局：   平  局")
                    break
                done = self.get_done(obs)  # 推演结束(分出胜负或达到最大时长)
                if done[0]:
                    obs = self.final_run_env(obs)
                else:
                    obs = self.run_env(obs)

            print("共", i + 1, "局：   红方胜", battle_results[0], "局：   蓝方胜", battle_results[1], "局：   平局",battle_results[2], "局")

            if (map_start == "Y" or map_start == "y") and i + 1 < num_episodes:
                self.end()
                print("请点击态势显示工具的实时态势按钮或者快捷键F2！")
                reset = input("是否重置运行? Y or N")
                print("重置运行:", reset)
                if not (reset == "Y" or reset == "y"):
                    return

            self._reset()
        return battle_results

    def run_env(self, obs):
        action = self.get_action(obs)
        return self.step(action)  # 环境推进一步

    def final_run_env(self, obs):
        actions = []
        cur_time = obs["sim_time"]
        for side, agent in self.agents.items():
            # cmd_list = self._agent_step(agent, cur_time, obs[side])
            if side == "red":
                cmd_list = agent.final_step(cur_time, obs[side])
            else:
                cmd_list = agent.step(cur_time, obs[side], obs)
            # print(cmd_list)
            actions.extend(cmd_list)

        return self.step(actions)


# 启动单个xsim,同时支持host模式
def main_signal():
    address = ADDRESS['ip'] + ":" + str(ADDRESS['port'])
    battle_runner = BattleRunnerSignal(address)
    episode_num = config["episode_time"]
    map_start = input("是否已启动态势显示工具? Y or N")
    battle_runner.run(episode_num, map_start)


def mult(address):
    battle_runner = BattleRunnerSignal(address)
    episode_num = config["episode_time"]
    battle_runner.run(episode_num)


# 启动多个xsim,不支持实时展示
def main_mult():
    po = Pool(XSIM_NUM)
    for i in range(XSIM_NUM):
        address = ADDRESS['ip'] + ":" + str(int(ADDRESS['port']) + i)
        po.apply_async(mult, (address,))
        print(f"第{i+1}个XSIM启动成功, address=‘{address}’!")

    po.close()
    po.join()


if __name__ == '__main__':
    if ISHOST:
        main_signal()
    else:
        main_mult()



