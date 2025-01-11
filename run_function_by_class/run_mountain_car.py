# -*- coding: utf-8 -*-
"""
@File    : run_mountain_car.py      # 文件名，run_mountain_car_game表示当前文件名
@Time    : 2025/1/10         # 创建时间，2025/1/10表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""

from envs.mountaincar import *
from run_function_by_class.run_select_func import run_select_func
def run_mountain_car():
    """
    出租车调度
    :return:
    """
    # 创建 Blackjack 环境
    env = MountainCar()
    # 策略评估并绘制价值函数图
    policy_name = {
        0: "施加持续向右的力",
        1: "隐藏SARSA策略更新",
        2: "期望SARSA策略更新",
        3: "Q-Learning更新",
        4: "Double-Q-Learning更新",
        5: "资格迹学习更新",
    }
    get_function = {
        0: env.play_game,  # 执行一步游戏

    }
    # 选择get_function中序号
    choice_method = 0
    run_select_func(get_function, choice_method)