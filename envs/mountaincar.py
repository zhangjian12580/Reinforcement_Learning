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

from pygame.transform import scale

from envs.env_template import Env
from tools.visualizer import Visualizer
from tools.save_policy import Policy_loader

logger = logging.getLogger(__name__)  # 使用当前模块名
from envs.global_set import *


class EnvInit(Env):
    """
    算法参数初始化
    """

    def __init__(self, name='MountainCar-v0', render_mode=render_model[0], render=True):
        super().__init__(name, render_mode, render)
        # 是否开启动画
        if render:
            self.env = gym.make(name, render_mode=render_mode)
        else:
            self.env = gym.make(name)

        self.render = render
        # 游戏轮数
        self.game_rounds = 2001
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
        self.train = False
        # 折扣因子，决定了未来奖励的影响
        self.gamma = 0.9
        # 学习率
        self.learning_rate = 0.1
        # 柯西收敛范围
        self.tolerant = 1e-6
        # ε-柔性策略因子
        self.epsilon = 0.01

    @property
    def print_env_info(self):
        return self.__env_info

    def __env_info(self):
        logger.info(f'观测空间：{self.envs.observation_space}')
        logger.info(f'动作空间：{self.envs.action_space}')
        logger.info(f'位置范围：{(self.envs.min_position, self.envs.max_position)}')
        logger.info(f'速度范围：{(-self.envs.max_speed, self.envs.max_speed)}')
        logger.info(f'目标位置：{self.envs.goal_position}')


class TileCoder:
    """
    瓦砖编码：对于可以实现目标函数的状态，尽可能捕捉相似性，对于无法完成目标函数的状态，尽可能区分差异性
    输入特征可能是：[位置，障碍物，目标距离，方向信息]，对应的权重也有相同维度，
    但是在具体分组时，会根据具体情况分配权重，比如障碍物，（1，-10，10，5），将整体价值拉低
    而在线性函数中会根据状态的总体不同情况分组，比如首先是障碍物，然后是目标距离...，不会对所有情况计算权重，
    这就实现了泛化，即相似特征可以使用相同权重

    本质：编码过程就是对真实世界物理量便于使用强化学习训练而将连续的状态转化为离散的表示，
    在定义后，所有训练的状态向量都应该遵循这个规则，在训练后，在将输出传递给现实世界进行决策规划，

    传感器采样得到的物理信息（例如位置、速度、角度等）会实时传递给瓦砖网络进行编码，
    瓦砖网络负责将这些连续的物理信息离散化，
    并通过特定的编码方式（例如瓦砖编码、哈希编码等）将这些信息转换为适合用于训练的特征表示
    """
    def __init__(self, layers, features):
        self.layers = layers  # 瓦砖的层数
        self.features = features  # 最多能够存储的特征数，权重参数的维度
        self.codebook = {}  # 用于存储每个编码对应的特征

    @property
    def get_features(self):
        return self.__get_features

    def __get_features(self, codeword):
        # codebook = {(0, 25, 10, 1): 0, (0, 25, 10, 2): 1, (0, 25, 11, 1): 2}
        if codeword in self.codebook:
            return self.codebook[codeword]  # 如果已经计算过这个编码，则返回对应的特征ID
        # 每次多个codeword，+1
        count = len(self.codebook)
        if count >= self.features:
            return hash(codeword) % self.features  # 如果特征数量超出最大限制，进行哈希映射，
            # 该hash将里面的tuple多个值计算出一个整数，再取模防止哈希碰撞
        else:
            self.codebook[codeword] = count  # 如果特征数量未超出限制，则为该编码分配一个新的特征ID
            return count

    def __call__(self, floats=(), ints=()):
        """
        floats: 浮动特征，离散化的连续输入特征, floats = (3.4, 1.2)

        # 创建 BrickNetwork 类的实例，假设层数为 3
        network = BrickNetwork(layers=3)
        # 调用实例，传入浮动特征 (位置、速度) 和整数特征 (例如动作)
        floats = (3.4, 1.2)  # 假设位置是 3.4，速度是 1.2
        ints = (0,)  # 假设整数特征是 0，可能代表某个动作
        # 使用 __call__ 方法（实际上是直接通过实例调用）得到离散化的特征
        features = network(floats=floats, ints=ints)
        """
        dim = len(floats)

        # 举例：对于输入为(0,10)的区间，如果被layers=3划分，且每个划分的偏移量不同，
        # 不同的使得每一层的瓦砖划分具有不同的精度和视角，因此增强了编码的表达能力。

        # 例如，假设层数
        # m = 3，我们可能会对位置特征
        # x的每一层使用不同的偏移量：
        # 第一层：位置x划分为[0, 3), [3, 6), [6, 9), [9, 10]
        # 第二层：位置x划分为[0, 2), [2, 5), [5, 8), [8, 10]
        # 第三层：位置x划分为[0, 1), [1, 4), [4, 7), [7, 10]
        # 可以把缩放看作是面积的放大，因为面积是x^2，当x缩放3倍，就是3x，面积就是3*3*x^2，所以，是对于某一个特征是f*layer*layer
        scales_floats = tuple(f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            # 1 + dim * i目的是为了在不同的层（layer）和特征（i）之间引入不同的偏移量。
            # 当 i = 0 时，偏移量是 1 + 3 * 0 = 1，这就相当于给第一个特征（比如位置）添加一个基本的偏移量 1。
            # 当 i = 1 时，偏移量是 1 + 3 * 1 = 4，这就相当于给第二个特征（比如速度）添加一个偏移量 4。
            # 当 i = 2 时，偏移量是 1 + 3 * 2 = 7，这就相当于给第三个特征（比如角度）添加一个偏移量 7。
            # 将每一层的离散化特征和整数特征（如状态或动作）一起拼接成一个 codeword
            codeword = ((layer,) +
                        # dim作用: 增大不同特征之间的区别防止特征的偏移量相互干扰；瓦砖编码的表达能力下降
                        tuple(int((f + (1 + dim * i) * layer) / self.layers)
                              for i, f in enumerate(scales_floats)) +
                        ints)
            # codeword = (0, 25, 10, 1)
            feature = self.__get_features(codeword)
            features.append(feature)
        return features


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
