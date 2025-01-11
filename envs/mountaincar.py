# -*- coding: utf-8 -*-
"""
@File    : mountaincar.py      # 文件名，mountaincar表示当前文件名
@Time    : 2025/1/10         # 创建时间，2025/1/10表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""

import time
import gym
import numpy as np
import logging

from envs.env_template import Env
from tools.visualizer import Visualizer
from tools.save_policy import Policy_loader
logger = logging.getLogger(__name__)  # 使用当前模块名
from envs.global_set import *

class EnvInit(Env):
    """
    算法参数初始化
    """
    def __init__(self, name = 'MountainCar-v0', render_mode = render_model[0], render = True):
        super().__init__(name, render_mode, render)
        # 是否开启动画
        if render:
            self.env = gym.make(name, render_mode=render_mode)
        else:
            self.env = gym.make(name)

        self.render = render
        # 游戏轮数
        self.game_rounds = 2001
        # 获取状态空间的大小（假设 FrozenLake 是一个网格地图，这里 nrow 和 ncol 可以直接得到）
        self.State_Num = self.envs.observation_space.shape
        # 获取动作空间的大小，即可选择的动作数量
        self.Action_Num = self.envs.action_space.shape
        # 位置
        self.positions = []
        # 速度
        self.velocities = []
        # 保存模型
        self.save_policy = False
        # 加载模型
        self.load_model = False
        # self.q_sa = np.zeros((self.State_Num, self.Action_Num))
        self.train = False
        # 初始化Q（s，a）表
        # if self.load_model:
        #     self.q_sa = Policy_loader.load_policy(class_name = self.__class__.__name__,
        #                                           method_name = "play_game_by_tracy.csv")
        # else:
        #     self.q_sa = np.zeros((self.State_Num, self.Action_Num))

        # 折扣因子，决定了未来奖励的影响
        self.gamma = 0.9
        # 学习率
        self.learning_rate = 0.1
        # 柯西收敛范围
        self.tolerant = 1e-6
        # ε-柔性策略因子
        self.epsilon = 0.01

    def print_env_info(self):
        logger.info(f'观测空间：{self.envs.observation_space}')
        logger.info(f'动作空间：{self.envs.action_space}')
        logger.info(f'位置范围：{(self.envs.min_position, self.envs.max_position)}')
        logger.info(f'速度范围：{(-self.envs.max_speed, self.envs.max_speed)}')
        logger.info(f'目标位置：{self.envs.goal_position}')

class MountainCar(EnvInit):
    def __init__(self):
        super().__init__()

    def play_game(self):
        """
        智能体推演
        :return:
        """
        self.print_env_info()
        observation, _ = self.reset()
        while True:
            self.positions.append(observation[0])
            self.velocities.append(observation[1])
            next_observation, reward, terminated, truncated, _ = self.step(2)
            done = terminated or truncated
            if done:
                break
            observation = next_observation

        if next_observation[0] > 0.5:
            logger.info("成功")
        else:
            logger.info("失败")

        Visualizer.plot_maintain_curve(self.positions, self.velocities)

