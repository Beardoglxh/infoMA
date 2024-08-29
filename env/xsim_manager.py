import subprocess
import logging
import os
import re
import config
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

from config import HostID
class XSimManager(object):
    def __init__(self, time_ratio: int, address: str, image_name: str = 'xsim5:v1.0'):
        self.xsim_time_ratio = time_ratio  # XSIM引擎加速比
        self.image_name = image_name  # 镜像名称
        self.address = self.__isaddress(address)  # ip
        self.port = self.address.split(":")[1]
        logging.info('当前引擎地址：{}'.format(self.address))
        self.domain_group = self.port
        self.xsim_run_num = self.port   # xsim环境运行编号(run_num)
        self.collect_num = 50
        self.docker_name = 'xsim_' + str(self.xsim_run_num)  # 启动的容器名称
        self.scenario_name = config.scenario_name
        self.__start_env()

    def __del__(self):
        logging.info(u"正在清理{}容器环境，请稍等……".format(self.docker_name))
        self.close_env()

    def __isaddress(self, address):
        """检查IP地址是否正确"""
        address_obj = re.search(r"^((2[0-4]\d|25[0-5]|[01]?\d\d?)\.){3}(2[0-4]\d|25[0-5]|[01]?\d\d?):\d+$", address)
        if not address_obj:
            raise AddressError("无效的IP地址，请检查后重新输入！")

        return address

    #-v /home/ds/soft/XSIMStudio/x64:/home/x64
    #/home/ds/soft/RTMData:/home/x64/RTMData
    def __start_env(self):
        if config.ISHOST:
            docker_run = "docker run --network host -itd --name {} -v /home/ds/soft/RTMData:/home/x64/RTMData -w /home/x64 {} python daemon_server.py {} {} {} {} {}"\
                .format(self.docker_name, self.image_name, self.xsim_time_ratio, self.port, HostID, self.collect_num, self.scenario_name)
        else:
            docker_run = "docker run -p {}:{} -itd --name {} -v /home/ds/soft/RTMData:/home/x64/RTMData -w /home/x64 {} python daemon_server.py {} {} {} {} {}"\
                .format(self.port, self.port, self.docker_name, self.image_name, self.xsim_time_ratio, self.port, self.domain_group, self.collect_num, self.scenario_name)
        logging.info(docker_run)
        # subprocess.Popen(docker_run, shell=True)
        os.system(docker_run)

    def close_env(self):
        # container_info = subprocess.getoutput(f"docker ps -a -f 'name={self.dokcer_name}' --format '{{{{.Status}}}}'")
        # pass
        logging.warning(self.docker_name)
        logging.info("docker ps -a -f 'name={}' --format '{{{{.Status}}}}'".format(self.docker_name))
        container_info = subprocess.getoutput("docker ps -a -f 'name={}' --format '{{{{.Status}}}}'".format(self.docker_name))
        logging.info(container_info)
        docker_cmd = ''
        if container_info is None:
            return

        if 'Exited' in container_info:
            docker_cmd = 'docker rm {}'.format(self.docker_name)
        elif 'Up' in container_info:
            docker_stop = 'docker stop {}'.format(self.docker_name)
            subprocess.call(docker_stop, shell=True,  timeout=30)
            docker_cmd = 'docker rm {}'.format(self.docker_name)
        subprocess.call(docker_cmd, shell=True,  timeout=30)


class AddressError(Exception):
    """IP地址无效异常"""
    pass