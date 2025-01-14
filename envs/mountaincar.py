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

    def __init__(self, name='MountainCar-v0', render_mode=render_model[0], render=False):
        super().__init__(name, render_mode, render)
        # 是否开启动画
        if render:
            self.env = gym.make(name, render_mode=render_mode)
        else:
            self.env = gym.make(name)

        self.render = render
        # 游戏轮数
        self.game_rounds = 300
        # self.State_Num = self.env.observation_space.n
        # 获取动作空间的大小，即可选择的动作数量
        self.Action_Num = self.env.action_space.n
        # 位置
        self.positions = []
        # 速度
        self.velocities = []
        # 保存模型
        self.save_policy = True
        # 加载模型
        self.load_model = True
        self.train = True
        # 折扣因子，决定了未来奖励的影响
        self.gamma = 1.
        # 学习率
        self.learning_rate = 0.03
        # 柯西收敛范围
        self.tolerant = 1e-6
        # ε-柔性策略因子
        self.epsilon = 0.001

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

    def __init__(self, layers, features, codebook=None):
        self.layers = layers  # 瓦砖的层数
        self.features = features  # 最多能够存储的特征数，权重参数的维度
        self.codebook = codebook if codebook else {}  # 用于存储每个编码对应的特征

    @property
    def get_features(self):
        return self.__get_features

    def __get_features(self, codeword):
        # codebook = {(0, 25, 10, 1): 0, (0, 25, 10, 2): 1, (0, 25, 11, 1): 2}
        logger.debug(f"codeword:{codeword}")
        codeword = tuple(codeword)
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
            # dim作用: 增大不同特征之间的区别防止特征的偏移量相互干扰；瓦砖编码的表达能力下降
            codeword = ((layer,) + tuple(int((f + (1 + dim * i) * layer) / self.layers)
                                         for i, f in enumerate(scales_floats)) +
                        (ints if isinstance(ints, tuple) else (ints,)))
            # codeword = (0, 25, 10, 1)
            feature = self.__get_features(codeword)
            features.append(feature)
        return features

class SARSAAgent(EnvInit):
    """
    函数近似SARSA算法
    1. 维度数量问题
    2. 状态编码过程
    3. 训练过程以及算法更新过程
    """
    def __init__(self, layers=8, features=1893):
        """
        初始化SARSA Agent
        :param layers: TileCoder的层数（多层编码用于更细粒度的状态表示）
        :param features: 总的特征数量
        """
        super().__init__(render=True)  # 初始化父类（包含环境相关参数）
        self.render = False  # 默认关闭渲染
        self.obs_low = self.env.observation_space.low  # 环境观测的最小值
        self.obs_scale = self.env.observation_space.high - self.env.observation_space.low  # 环境观测的范围
        self.layers = layers  # TileCoder 的层数
        self.features = features  # 特征数量

        if not self.load_model:  # 如果未加载模型，则初始化 TileCoder 和权重
            self.encoder = TileCoder(layers, features)  # 初始化TileCoder，用于状态和动作的编码
            self.w = np.zeros(features)  # 初始化权重为零向量
        else:  # 如果加载模型，则恢复权重和编码器状态
            self.w, codebook = Policy_loader.load_w_para(class_name=self.__class__.__name__,
                                                         method_name="play_game_by_sarsa_resemble.pkl")
            self.encoder = TileCoder(layers, features, codebook)  # 使用加载的codebook初始化TileCoder

    def encode(self, observation, action):
        """
        编码观测和动作为特征向量
        :param observation: 当前状态（连续值）
        :param action: 动作（离散值）
        :return: 特征索引列表
        """
        # 将观测值归一化到 [0, 1] 范围，并转换为元组
        states = tuple((observation - self.obs_low) / self.obs_scale)
        # 将动作封装为元组
        actions = (action,)
        # 使用TileCoder编码为特征索引
        return self.encoder(states, actions)

    def get_q(self, observation, action):
        """
        获取动作的Q值
        :param observation: 当前状态
        :param action: 动作
        :return: 对应的Q值
        """
        features = self.encode(observation, action)  # 编码观测和动作为特征索引
        return self.w[features].sum()  # 根据权重和特征计算Q值

    def agent_resemble_decide(self, observation):
        """
        根据当前策略进行动作决策
        :param observation: 当前状态
        :return: 选定的动作
        """
        if np.random.rand() < self.epsilon:  # 以epsilon概率随机选择动作（探索）
            return np.random.randint(self.Action_Num)
        else:  # 否则选择Q值最大的动作（利用）
            qs = [self.get_q(observation, action) for action in range(self.Action_Num)]
            return np.argmax(qs)  # 返回Q值最大的动作索引

    def sarsa_resemble_learn(self, observation, action, reward, next_observation, next_action, done):
        """
        使用SARSA更新规则进行学习
        :param observation: 当前状态
        :param action: 当前动作
        :param reward: 当前奖励
        :param next_observation: 下一个状态
        :param next_action: 下一个动作
        :param done: 是否为终止状态
        """
        # 计算目标值u
        u = reward + (1. - done) * self.gamma * self.get_q(next_observation, next_action)
        # TD误差：目标值与当前Q值的差
        td_error = u - self.get_q(observation, action)
        # 获取当前状态和动作的特征索引
        features = self.encode(observation, action)
        # 根据TD误差更新权重
        self.w[features] += self.learning_rate * td_error

    def play_game_by_sarsa_resemble(self, train=False):
        """
        使用SARSA算法训练
        :param train:
        :return:
        """
        episode_reward = 0
        observation, _ = self.reset()
        action = self.agent_resemble_decide(observation)

        done = False
        while True:
            if self.render:
                self.env.render()

            next_observation, reward, terminated, truncated, _ = self.step(action)
            episode_reward += reward
            # logger.info(f"当前状态:{next_observation}")
            next_action = self.agent_resemble_decide(next_observation)
            # logger.info(f"选择动作:{next_action}")

            if terminated or truncated:
                done = True

            if train:
                self.sarsa_resemble_learn(observation, action, reward, next_observation, next_action, done)
            else:
                time.sleep(2)

            if done:
                logger.info(f"结束一轮游戏")
                break
            observation, action = next_observation, next_action
        return episode_reward


class SARSALamdaAgent(EnvInit):
    """
    函数近似SARSA(𝜆)算法
    """
    def __init__(self, lamda=0.9, layers=8, features=1893):
        super().__init__(render = True)
        self.lamda = lamda
        self.layers = layers
        self.features = features
        self.obs_low = self.env.observation_space.low
        self.obs_scale = self.observation_space.high - self.env.observation_space.low
        self.z = np.zeros(features)
        # 加载模型
        self.train = True
        self.load_model = True
        if not self.load_model:
            self.encoder = TileCoder(self.layers, self.features)
            self.w = np.zeros(features)
        else:
            self.w, codebook = Policy_loader.load_w_para(class_name=self.__class__.__name__,
                                                         method_name="play_game_by_sarsa_lamda.pkl")
            self.encoder = TileCoder(self.layers, self.features, codebook)

    def encode(self, observation, action):
        """
        编码
        """
        states = tuple((observation - self.obs_low) / self.obs_scale)
        actions = (action,)
        return self.encoder(states, actions)

    def get_q(self, observation, action):
        """
        获取动作价值
        :param observation:
        :param action:
        :return:
        """
        features = self.encode(observation, action)
        # logger.info(f"features:{features}")
        return self.w[features].sum()

    def agent_resemble_decide(self, observation):
        """
        决策
        :param observation:
        :return:
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Action_Num)
        else:
            qs = [self.get_q(observation, action) for action in range(self.Action_Num)]
            return np.argmax(qs)

    def SARSA_Lamda_learn(self, observation, action, reward, next_observation, next_action, done):
        u = reward
        if not done:
            u += (self.gamma * self.get_q(next_observation, next_action))
            self.z *= (self.gamma * self.lamda)
            features = self.encode(observation, action)
            self.z[features] = 1.
        td_error = u - self.get_q(observation, action)
        self.w += (self.learning_rate * td_error * self.z)
        if done:
            self.z = np.zeros_like(self.z)

    def play_game_by_sarsa_lamda(self, train=False):
        """
        使用SARSA算法训练
        :param train:
        :return:
        """
        episode_reward = 0
        observation, _ = self.reset()
        action = self.agent_resemble_decide(observation)
        done = False
        while True:
            if self.render:
                self.env.render()

            next_observation, reward, terminated, truncated, _ = self.step(action)
            # taxi_row, taxi_col, pass_loc, dest_idx = self.envs.decode(next_observation)

            # if not train:
            #     logger.info(f"下一个状态：{(taxi_row, taxi_col)}")
            episode_reward += reward

            next_action = self.agent_resemble_decide(next_observation)

            # if not train:
            #     logger.info(f"下一个动作：{self.translate(action)}")

            if terminated or truncated:
                done = True

            if train:
                self.SARSA_Lamda_learn(observation, action, reward, next_observation, next_action, done)
            else:
                time.sleep(2)

            if done:
                logger.info(f"结束一轮游戏")
                break
            observation, action = next_observation, next_action
        return episode_reward


class MountainCar(SARSAAgent, SARSALamdaAgent):
    def __init__(self):
        SARSAAgent.__init__(self)
        SARSALamdaAgent.__init__(self)
        self.class_name = self.__class__.__name__

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

    def game_iteration(self, show_policy):
        """
        迭代
        :param show_policy: 使用的更新策略方式
        """
        episode_reward = 0.
        episode_rewards = []  # 总轮数的奖励(某轮总奖励)列表

        method_name = "default"

        for game_round in range(1, self.game_rounds):
            logger.info(f"---第{game_round}轮训练---")
            if show_policy == "函数近似SARSA算法":
                # logger.info(f"函数近似SARSA算法")
                episode_reward = self.play_game_by_sarsa_resemble(train=True)  # 第round轮次的累积reward
                method_name = self.play_game_by_sarsa_resemble.__name__
            if show_policy == "函数近似SARSA(𝜆)算法":
                # logger.info(f"函数近似SARSA(𝜆)算法")
                episode_reward = self.play_game_by_sarsa_lamda(train=True)  # 第round轮次的累积reward
                method_name = self.play_game_by_sarsa_lamda.__name__

            if self.save_policy:
                save_data = {
                    "weights": self.w,
                    "encoder": self.encoder.codebook if self.encoder else None
                }
                Policy_loader.save_policy(method_name, self.class_name, save_data)

            if episode_reward is not None:
                episode_rewards.append(episode_reward)
                logger.info(f"第{game_round}轮奖励: {episode_reward}")
            else:
                logger.warning(f"第{game_round}轮奖励为 None，已跳过。")

            Visualizer.plot_cumulative_avg_rewards(episode_rewards, game_round, self.game_rounds, self.class_name,
                                                   method_name)

        print(
            f"平均奖励：{(np.round(np.mean(episode_rewards), 2))} = {np.sum(episode_rewards)} / {len(episode_rewards)}")
        print(
            f"最后100轮奖励：{(np.round(np.mean(episode_rewards[-500:]), 2))} = {np.sum(episode_rewards[-500:])} / {len(episode_rewards[-500:])}")
