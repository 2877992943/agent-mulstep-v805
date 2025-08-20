
"""

env
agent
 run

"""
import os,json
import logging
import datetime
import time

from uitars_agent_428 import UITARSAgent
from env_uos import Env
agent=UITARSAgent()
example_result_dir='./tmp2'
env=Env(screenshot_dir=example_result_dir)
# Configure logging
logging.basicConfig(filename='./env_log.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


from run_example1 import run_1example


if __name__=='__main__':
    tasks="""
如何打开控制中心？
如何更新或升级系统？
如何设置个人热点？
如何查看 IP 地址和 MAC 信息？
如何更改系统字体及大小？
如何将自定义壁纸设置为自动切换的壁纸？
如何设置密码有效期?
如何开启或关闭系统提示音?
如何切换系统语言?
如何修改帐户密码?
如何显示或隐藏任务栏?
如何查看系统字体和用户字体包
如何查询计算机的详细配置
如何设置全局搜索范围
如何自定义锁屏时间""".strip().split('\n')
    for ix,inst in enumerate(tasks):
        if ix in [0,1,2,3,4,5,6,7,8,9,10,11]:continue
        time.sleep(2)
        inst=inst.strip()
        if not inst:continue
        inst=inst.strip().replace('如何','').replace('?','').replace('？','')
        run_1example(inst,name=ix)
        break