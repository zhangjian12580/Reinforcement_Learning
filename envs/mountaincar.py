# -*- coding: utf-8 -*-
"""
@File    : mountaincar.py      # 文件名，mountaincar表示当前文件名
@Time    : 2025/1/10         # 创建时间，2025/1/10表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""

import time
from collections import deque
import torch
import os
import keras
from keras.api.initializers import GlorotUniform
import tensorflow as tf
import gym
import numpy as np
import logging
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from envs.env_template import Env
from tools.visualizer import Visualizer
from tools.save_policy import Policy_loader

logger = logging.getLogger(__name__)  # 使用当前模块名
from envs.global_set import *

# 创建全局变量模拟学习阶段
"""
小车上山环境说明：
以水平方向为参考，位置速度范围都是以山底的为0，
向左走的时候，速度为负，位置为负-[-1.2,0.6]
向右走的时候，速度为正，位置为正-[-0.07,0.07]
动作范围施力方向：向左，无，向右-[0,1,2]
初始位置为[-0.6,-0.4]
初始速度：0
"""


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
        self.game_rounds = 25000
        # 获取动作空间的大小，即可选择的动作数量
        self.Action_Num = self.env.action_space.n
        # 位置
        self.positions = []
        # 速度
        self.velocities = []
        # 保存模型
        self.save_policy = False
        # 加载模型
        self.load_model = True
        # 折扣因子，决定了未来奖励的影响
        self.gamma = 1.
        # 学习率
        self.learning_rate = 0.01
        # 柯西收敛范围
        self.tolerant = 1e-6
        # ε-柔性策略因子
        self.epsilon = 0.001
        self.translate_action = {
            0: "左",
            1: "无",
            2: "右"
        }

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
        self.codebook = codebook if codebook else {}  # 用于存储每个编码对应的特征{(0,3,2,3):1,(0,1,2,1):2}

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
            # feature在self.codebook中对应的值，这个映射相当于做个转换，将编码元组转换为w中的索引
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
        super().__init__()  # 初始化父类（包含环境相关参数）
        self.obs_low = self.env.observation_space.low  # 环境观测的最小值
        self.obs_scale = self.env.observation_space.high - self.env.observation_space.low  # 环境观测的范围
        self.layers = layers  # TileCoder 的层数
        self.features = features  # 特征数量

        if not self.load_model:  # 如果未加载模型，则初始化 TileCoder 和权重
            self.tile_coder = TileCoder(layers, features)  # 初始化TileCoder，用于状态和动作的编码
            self.weights = np.zeros(features)  # 初始化权重为零向量，把weights看作是Q-table
        else:  # 如果加载模型，则恢复权重和编码器状态
            self.weights, codebook = Policy_loader.load_w_para(class_name=self.__class__.__name__,
                                                               method_name="play_game_by_sarsa_resemble.pkl")
            self.tile_coder = TileCoder(layers, features, codebook)  # 使用加载的codebook初始化TileCoder

    def preprocess_encode(self, observation, action):
        """
        编码观测和动作为特征向量
        :param observation: 当前状态（连续值）
        :param action: 动作（离散值）
        :return: 特征索引列表[]->list
        """
        # 将观测值归一化到 [0, 1] 范围，并转换为元组
        states = tuple((observation - self.obs_low) / self.obs_scale)
        # 将动作封装为元组
        actions = (action,)
        # 使用TileCoder编码为特征索引，该瓦砖网络将（states，actions）转换为特征索引（在weights中的索引）
        return self.tile_coder(states, actions)

    def get_weights(self, observation, action):
        """
        根据layers层数获取当前的（observation, action）在不同层的特征的索引，通过索引在w中找到参数，求和
        获取动作的Q值
        :param observation: 当前状态
        :param action: 动作
        :return: 对应的weights或者Q(s, a)值
        """
        features = self.preprocess_encode(observation, action)  # 编码观测和动作为特征索引: [16, 29, 71, 19, 20, 21, 22, 23]
        return self.weights[features].sum()  # 根据权重和特征计算Q值

    def agent_resemble_decide(self, observation):
        """
        根据当前策略进行动作决策
        :param observation: 当前状态
        :return: 选定的动作
        """
        if np.random.rand() < self.epsilon:  # 以epsilon概率随机选择动作（探索）
            return np.random.randint(self.Action_Num)
        else:  # 否则选择Q值最大的动作（利用）
            q_value = [self.get_weights(observation, action) for action in range(self.Action_Num)]
            return np.argmax(q_value)  # 返回Q值最大的动作索引

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
        u_t = reward + (1. - done) * self.gamma * self.get_weights(next_observation, next_action)
        # TD误差：目标值与当前Q值的差
        td_error = u_t - self.get_weights(observation, action)
        # 获取当前状态和动作的特征索引
        features = self.preprocess_encode(observation, action)
        # 根据TD误差更新权重
        self.weights[features] += self.learning_rate * td_error

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
        super().__init__()
        self.lamda = lamda
        self.layers = layers
        self.features = features
        self.obs_low = self.env.observation_space.low
        self.obs_scale = self.observation_space.high - self.env.observation_space.low
        self.e_tracy = np.zeros(features)
        # 加载模型
        self.train = True
        self.load_model = True
        if not self.load_model:
            self.tile_coder = TileCoder(self.layers, self.features)
            self.weights = np.zeros(features)
        else:
            self.weights, codebook = Policy_loader.load_w_para(class_name=self.__class__.__name__,
                                                               method_name="play_game_by_sarsa_lamda.pkl")
            self.tile_coder = TileCoder(self.layers, self.features, codebook)

    def process_encode(self, observation, action):
        """
        编码
        """
        states = tuple((observation - self.obs_low) / self.obs_scale)
        actions = (action,)
        return self.tile_coder(states, actions)

    def get_weights(self, observation, action):
        """
        获取动作价值
        :param observation:
        :param action:
        :return:
        """
        features = self.process_encode(observation, action)
        # logger.info(f"features:{features}")
        return self.weights[features].sum()

    def agent_resemble_decide(self, observation):
        """
        决策
        :param observation:
        :return:
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Action_Num)
        else:
            q_value = [self.get_weights(observation, action) for action in range(self.Action_Num)]
            return np.argmax(q_value)

    def sarsa_lamda_learn(self, observation, action, reward, next_observation, next_action, done):
        """
        使用 SARSA(λ) 算法进行学习
        在 SARSA(λ) 或其他基于资格迹的算法中，所有的权重都会被更新，但是每个权重的更新幅度是不同的，
        具体取决于它们对应的状态-动作对在学习过程中的 资格迹（eligibility trace）。
        资格迹反映了每个状态-动作对对当前误差的贡献程度，也就是它在历史中被访问的频率。
        资格迹越大的状态-动作对，会得到更多的更新，因为它们在历史中对当前回报的影响更大。

        :param observation: 当前状态
        :param action: 当前动作
        :param reward: 当前奖励
        :param next_observation: 下一个状态
        :param next_action: 下一个动作
        :param done: 是否为终止状态
        """

        # 计算当前的目标值 u_t
        u_t = reward  # 当前奖励作为初始目标值
        if not done:
            # 如果不是终止状态，目标值中需要加入下一状态-动作对的折扣 Q 值
            u_t += (self.gamma * self.get_weights(next_observation, next_action))

        # 减小当前迹线的强度 (递减因子由 gamma 和 λ 共同决定)
        self.e_tracy *= (self.gamma * self.lamda)

        # 根据当前状态和动作获取对应特征索引
        features = self.process_encode(observation, action)

        # 将当前状态-动作对应的特征索引的迹线值设置为 1
        # 这表示最近访问的状态-动作对有最高的更新优先级
        self.e_tracy[features] = 1.

        # 计算 TD 误差 (Temporal Difference Error)
        td_error = u_t - self.get_weights(observation, action)

        # 根据 TD 误差以及迹线值更新所有权重
        # 迹线值表示历史上状态-动作对对当前学习过程的影响程度
        self.weights += (self.learning_rate * td_error * self.e_tracy)

        # 如果是终止状态，将迹线值重置为零
        if done:
            self.e_tracy = np.zeros_like(self.e_tracy)

    def play_game_by_sarsa_lamda(self, train=False):
        """
        使用SARSA算法训练
        :param train:
        :return:
        """
        episode_reward = 0
        observation, _ = self.reset()  # observation：[位置， 速度]
        action = self.agent_resemble_decide(observation)
        done = False
        while True:
            if self.render:
                self.env.render()

            next_observation, reward, terminated, truncated, _ = self.step(action)

            if not train:
                logger.info(f"下一个状态：{next_observation}")
            episode_reward += reward

            next_action = self.agent_resemble_decide(next_observation)

            if not train:
                logger.info(f"下一个动作：{self.translate_action[action]}")

            if terminated or truncated:
                done = True

            if train:
                self.sarsa_lamda_learn(observation, action, reward, next_observation, next_action, done)
            else:
                time.sleep(2)

            if done:
                logger.info(f"结束一轮游戏")
                break
            observation, action = next_observation, next_action
        return episode_reward


class DQNReplayer:
    """
    经验回放类，用于存储和采样 DQN 中的经验。

    经验回放池（Replay Buffer）用于存储智能体在与环境交互过程中生成的经验（状态、动作、奖励、下一个状态、是否终止）。这些经验随后用于训练神经网络，以使模型学习到最佳策略。
    """

    def __init__(self, capacity):
        """
        初始化经验回放池。

        参数：
        - capacity: int，回放池的容量，决定了最多能存储多少条经验。
        """
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation',  # 当前状态
                                            'action',  # 执行动作
                                            'reward',  # 收到的奖励
                                            'next_observation',  # 下一状态
                                            'done'])  # 是否终止标志
        self.index = 0  # 当前存储位置的索引
        self.count = 0  # 当前回放池中存储的经验条数
        self.capacity = capacity  # 回放池的最大容量

    def replay_store(self, *args, pbar=None):
        """
        将新的经验存储到回放池。

        参数：
        - args: 包含一条经验的五个元素（当前状态、动作、奖励、下一个状态、是否终止）
        """
        # 存储经验
        self.memory.loc[self.index] = args
        # 更新存储位置的索引
        self.index = (self.index + 1) % self.capacity
        # 增加回放池中存储的经验条数，并确保不超过最大容量
        self.count = min(self.count + 1, self.capacity)

        # 如果进度条存在，更新进度条
        if pbar is not None:
            pbar.update(1)

    def replay_sample(self, size):
        """
        从经验回放池中随机采样一批经验。

        参数：
        - size: int，要采样的经验数量

        """
        # 从存储的经验中随机选择索引
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


from torch.utils.tensorboard import SummaryWriter


class DQNAgentTorch(EnvInit):
    """
    Deep Q-learning Network with PyTorch
    """

    def __init__(self,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 replayer_capacity: int = 10000,
                 batch_size: int = 64):
        super().__init__()
        # 超参数设置
        net_kwargs = {'hidden_sizes': [64, ]}  # 神经网络隐藏层设置
        self.Action_Num = self.env.action_space.n  # 动作空间的维度
        observation_dim = self.env.observation_space.shape[0]  # 状态空间维度
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索概率

        # TensorBoard writer 用于记录训练日志
        current_time = time.localtime()
        log_dir = time.strftime("runs/dqn_torch/%Y_%m_%d_%H_%M", current_time)
        self.writer = SummaryWriter(log_dir=log_dir)

        # 其他超参数
        self.learn_step_counter = int(0)  # 学习步计数器
        self.learning_rate = 0.001 # 学习率
        self.goal_position = 0.5
        self.batch_size = batch_size # # 表示每次训练从数据集中提取 batch_size 个样本
        self.replay_start_size = 1000  # 经验池开始训练所需的最小样本数量
        self.update_lr_steps = 10000 # 学习率刷新间隔

        # 用于跟踪最近游戏的完成率
        self.done_rate = deque(maxlen=100)

        # 初始化经验池
        self.replayer = DQNReplayer(replayer_capacity)

        # 初始化评估网络和目标网络
        self.evaluate_net_pytorch = self.build_torch_network(input_size=observation_dim,
                                                             output_size=self.Action_Num, **net_kwargs)
        self.target_net_pytorch = self.build_torch_network(input_size=observation_dim,
                                                           output_size=self.Action_Num, **net_kwargs)

        # 优化器
        self.dqn_optimizer = torch.optim.Adam(self.evaluate_net_pytorch.parameters(), lr=self.learning_rate)

        # 如果加载模型
        if self.load_model:
            checkpoint = torch.load("tools/policy_dir/MountainCar/evaluate_net_pytorch.pth", weights_only=True)
            self.evaluate_net_pytorch.load_state_dict(checkpoint["model_state_dict"])
            self.dqn_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->evaluate_net_pytorch")

        # 将评估网络的权重复制到目标网络
        self.target_net_pytorch.load_state_dict(self.evaluate_net_pytorch.state_dict())

    def build_torch_network(self,
                            input_size,
                            hidden_sizes,
                            output_size,
                            activation=nn.ReLU,
                            output_activation=None):
        """
        构建简单的前馈神经网络
        """
        layers = []
        input_dim = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))  # 全连接层
            layers.append(activation())  # 激活函数
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, output_size))  # 输出层

        if output_activation:
            layers.append(output_activation())

        model = nn.Sequential(*layers)  # 顺序模型
        return model

    def dqn_torch_agent_learn(self, observation, action, reward, next_observation, done):
        """
        使用 DQN 算法更新网络
        """
        self.evaluate_net_pytorch.train()  # 切换到训练模式

        # 如果经验池样本不足，进行加载
        if self.replayer.count <= self.replay_start_size:
            with tqdm(total=10000, initial=self.replayer.count, dynamic_ncols=True, desc="经验池加载进度") as pbar:
                for _ in range(10000):
                    self.replayer.replay_store(observation, action, reward, next_observation, done, pbar=pbar)
                    time.sleep(0.0002)  # 模拟加载延迟

        # 存储经验并采样
        self.replayer.replay_store(observation, action, reward, next_observation, done)
        observations, actions, rewards, next_observations, dones = self.replayer.replay_sample(self.batch_size)

        # 转换为 PyTorch 张量
        observations = torch.tensor(observations, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_observations = torch.tensor(next_observations, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 目标网络计算 Q 值
        next_qs = self.target_net_pytorch(next_observations).detach()
        next_max_qs = next_qs.max(dim=-1)[0]
        us = rewards + self.gamma * next_max_qs * (1. - dones)

        # 当前 Q 值计算
        qs = self.evaluate_net_pytorch(observations)
        targets = qs.clone()
        targets[torch.arange(self.batch_size), actions] = us

        # 损失函数计算
        loss = nn.SmoothL1Loss()(qs, targets)

        # 记录损失和平均 Q 值
        if self.learn_step_counter % 50 == 0:
            self.writer.add_scalar("Loss/train", loss.item(), self.learn_step_counter)
            avg_q_value = qs.mean().item()
            self.writer.add_scalar("Q Value/Average", avg_q_value, self.learn_step_counter)

        # 反向传播更新权重
        self.evaluate_net_pytorch.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

    def dqn_torch_agent_decide(self, observation):
        """
        根据当前状态选择动作
        """
        if np.random.rand() < self.epsilon:  # 进行随机探索
            return np.random.randint(self.Action_Num)

        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        qs = self.evaluate_net_pytorch(observation)
        return qs.argmax(dim=1).item()

    def play_game_by_dqn_torch_learning(self, train=False):
        """
        使用 DQN 算法训练和评估
        :param train: 是否训练模式
        :return: 累积奖励
        """
        episode_reward = 0
        observation, _ = self.reset()
        done = False

        if not train:
            logger.info(f"****启动评估阶段****")
            self.evaluate_net_pytorch.eval()
            self.target_net_pytorch.eval()

        while True:
            if self.render:
                self.env.render()

            if not train:
                with torch.no_grad():
                    action = self.dqn_torch_agent_decide(observation)
            else:
                action = self.dqn_torch_agent_decide(observation)

            next_observation, reward, terminated, truncated, _ = self.step(action)
            episode_reward += reward

            if terminated or truncated:
                done = True
                self.learn_step_counter += 1

            if train:
                self.dqn_torch_agent_learn(observation, action, reward, next_observation, done)

            if done:
                logger.info(f"结束一轮游戏")
                flag = True if episode_reward > -200 else False
                self.done_rate.append(flag)
                if train:
                    if self.learn_step_counter % 2000 == 0:
                        self.epsilon = max(0.01, self.epsilon * 0.995)
                    if self.learn_step_counter and self.learn_step_counter % 100 == 0:
                        self.target_net_pytorch.load_state_dict(self.evaluate_net_pytorch.state_dict())
                break

            observation = next_observation
        return episode_reward

    def refresh_writer(self, step):
        """
        刷新 TensorBoard Writer
        """
        self.writer.close()
        current_time = time.localtime()
        log_dir = time.strftime("runs/dqn_torch/%Y_%m_%d_%H_%M", current_time)
        new_log_dir = f"{log_dir}/{step}"
        self.writer = SummaryWriter(log_dir=new_log_dir)

    def close(self):
        """
        关闭 TensorBoard SummaryWriter
        """
        self.writer.close()
        logger.info("TensorBoard SummaryWriter 已关闭")


class DoubleDQNAgent(EnvInit):

    def __init__(self,
                 gamma: float = 0.99,
                 epsilon: float = 0.01,
                 replayer_capacity: int = 10000,
                 batch_size: int = 64):
        super().__init__()
        net_kwargs = {'hidden_sizes': [64, ]}
        self.Action_Num = self.env.action_space.n
        observation_dim = self.env.observation_space.shape[0]
        self.gamma = gamma
        self.epsilon = epsilon  #
        # TensorBoard writer
        current_time = time.localtime()
        log_dir = time.strftime("runs/double_dqn_torch/%Y_%m_%d_%H_%M", current_time)
        self.ddqn_writer = SummaryWriter(log_dir=log_dir)
        self.ddqn_learn_step_counter = int(0)  # 学习步数计数器
        self.ddqn_learning_rate = 0.0001
        self.ddqn_batch_size = batch_size
        self.ddqn_replay_start_size = 1000
        self.ddqn_training_started = False
        self.done_rate = deque(maxlen=100)
        self.done_rate.clear()
        self.ddqn_replayer = DQNReplayer(replayer_capacity)
        self.ddqn_evaluate_net_pytorch = self.ddqn_build_torch_network(input_size=observation_dim,
                                                                       output_size=self.Action_Num, **net_kwargs)
        self.ddqn_target_net_pytorch = self.ddqn_build_torch_network(input_size=observation_dim,
                                                                     output_size=self.Action_Num, **net_kwargs)
        self.ddqn_optimizer = torch.optim.Adam(self.ddqn_evaluate_net_pytorch.parameters(), lr=self.ddqn_learning_rate)

        if self.load_model:
            # 加载前保存模型参数
            # logger.info(f"initial_state_dict:{initial_state_dict}")
            checkpoint = torch.load("tools/policy_dir/MountainCar/ddqn_evaluate_net_pytorch.pth", weights_only=True)
            self.ddqn_evaluate_net_pytorch.load_state_dict(checkpoint["model_state_dict"])

            # logger.info(f"evaluate_net_pytorch:{self.evaluate_net_pytorch.state_dict()}")
            self.ddqn_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # logger.info(f"dqn_optimizer:{self.dqn_optimizer}")
            logger.info(f"成功加载--->evaluate_net_pytorch")

        self.ddqn_target_net_pytorch.load_state_dict(self.ddqn_evaluate_net_pytorch.state_dict())

    def ddqn_build_torch_network(self,
                                 input_size,
                                 hidden_sizes,
                                 output_size,
                                 activation=nn.ReLU,
                                 output_activation=None):
        """
        Build a simple feed-forward neural network with PyTorch
        """
        layers = []
        input_dim = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation())
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, output_size))

        if output_activation:
            layers.append(output_activation())

        model = nn.Sequential(*layers)
        return model

    def ddqn_torch_agent_decide(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.Action_Num)

        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        qs = self.ddqn_evaluate_net_pytorch(observation)
        return qs.argmax(dim=1).item()

    def double_dqn_agent_learn(self, observation, action, reward, next_observation, done):
        self.ddqn_evaluate_net_pytorch.train()  # 切换到训练模式

        # 初始化进度条
        if self.ddqn_replayer.count <= self.ddqn_replay_start_size:
            # logger.info(self.replayer.index)
            with tqdm(total=10000, initial=self.ddqn_replayer.count, dynamic_ncols=True, desc="经验池加载进度") as pbar:
                # 假设我们不断存储经验
                for _ in range(10000):
                    # 存储经验并更新进度条
                    self.ddqn_replayer.replay_store(observation, action, reward, next_observation, done, pbar=pbar)
                    time.sleep(0.0002)

        self.ddqn_replayer.replay_store(observation, action, reward, next_observation, done)
        observations, actions, rewards, next_observations, dones = self.ddqn_replayer.replay_sample(
            self.ddqn_batch_size)

        # Convert numpy arrays to PyTorch tensors
        observations = torch.tensor(observations, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_observations = torch.tensor(next_observations, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 1. 计算当前网络（评估网络）在 next_observations 上的 Q 值
        next_eval_qs = self.ddqn_evaluate_net_pytorch(next_observations)

        # 2. 获取 next_eval_qs 中的最大 Q 值的索引作为选定的动作
        next_actions = next_eval_qs.argmax(dim=-1)  # `argmax` 用于沿着指定的维度找到最大值的索引

        # 3. 计算目标网络（target_net）在 next_observations 上的 Q 值
        next_qs = self.ddqn_target_net_pytorch(next_observations).detach()  # 使用目标网络并且 `detach()` 防止梯度回传

        # 4. 获取目标网络输出的每个样本的最大 Q 值（用于计算 Q-learning 的目标值）
        next_max_qs = next_qs.gather(dim=-1, index=next_actions.unsqueeze(-1))  # gather 提取每个样本对应的最大 Q 值
        next_max_qs = next_max_qs.squeeze(-1)  # 移除最后的维度，使其保持正确的形状

        # Q values from target network
        # next_qs = self.target_net_pytorch(next_observations).detach()
        # next_max_qs = next_qs.max(dim=-1)[0]
        us = rewards + self.gamma * next_max_qs * (1. - dones)

        # Get current Q values
        qs = self.ddqn_evaluate_net_pytorch(observations)

        # Update the Q-values for the taken actions
        targets = qs.clone()
        targets[torch.arange(self.ddqn_batch_size), actions] = us

        # Compute loss
        # loss = nn.MSELoss()(qs, targets)
        loss = nn.SmoothL1Loss()(qs, targets)

        if self.ddqn_learn_step_counter % 50 == 0:  # 每 50 步记录一次
            self.ddqn_writer.add_scalar("Loss/train", loss.item(), self.ddqn_learn_step_counter)
            avg_q_value = qs.mean().item()
            self.ddqn_writer.add_scalar("Q Value/Average", avg_q_value, self.ddqn_learn_step_counter)

        # Back_propagate
        self.ddqn_evaluate_net_pytorch.zero_grad()
        loss.backward()
        # Update weights using optimizer
        self.ddqn_optimizer.step()

    def play_game_by_double_dqn_torch_learning(self, train=False):
        """
        使用Q-Learning算法训练
        :param train: 是否是训练模式
        :return: 某一轮累积奖励
        """
        episode_reward = 0
        observation, _ = self.reset()
        done = False
        # 在推理阶段禁用梯度计算
        if not train:  # 只有在评估阶段才进行推理
            logger.info(f"****启动评估阶段****")
            self.ddqn_evaluate_net_pytorch.eval()  # 切换模型到评估模式
            self.ddqn_target_net_pytorch.eval()  # 切换目标网络到评估模式

        while True:
            if self.render:
                self.env.render()

            # 在推理阶段禁用梯度计算
            if not train:  # 只有在评估阶段才进行推理
                with torch.no_grad():  # 禁用梯度计算，避免不必要的内存使用
                    # start = time.time()
                    action = self.ddqn_torch_agent_decide(observation)  # 推理决策
                    # end = time.time()
                    # logger.info(f"选择-->{action}---{int(end-start)}秒")
            else:
                action = self.ddqn_torch_agent_decide(observation)

            next_observation, reward, terminated, truncated, _ = self.step(action)
            # if not train:  # 只有在评估阶段才进行推理
            # logger.info(f"状态-->{next_observation}")
            # logger.info(f"奖励-->{reward}")

            episode_reward += reward

            if terminated or truncated:
                done = True
                self.ddqn_learn_step_counter += 1

            if train:
                self.double_dqn_agent_learn(observation, action, reward, next_observation, done)
            else:
                time.sleep(0)

            if done:
                logger.info(f"结束一轮游戏")
                flag = True if episode_reward > -200 else False
                self.done_rate.append(flag)
                if train:
                    # if self.ddqn_learn_step_counter % 2500 == 0:  # 每 1000 步刷新一次
                    #     self.ddqn_refresh_writer(self.ddqn_learn_step_counter)
                    if self.ddqn_learn_step_counter % 2000 == 0:  # 每 1000 步刷新一次
                        self.epsilon = max(0.01, self.epsilon * 0.995)  # 每次减少，最低为 0.01
                    if self.ddqn_learn_step_counter and self.ddqn_learn_step_counter % 100 == 0:
                        self.ddqn_target_net_pytorch.load_state_dict(self.ddqn_evaluate_net_pytorch.state_dict())
                break

            observation = next_observation
        return episode_reward

    def ddqn_refresh_writer(self, step):
        self.ddqn_writer.close()  # 关闭旧的 writer
        new_log_dir = f"runs/double_dqn_torch_{step}"
        self.ddqn_writer = SummaryWriter(log_dir=new_log_dir)


class MountainCar(SARSAAgent, SARSALamdaAgent, DQNAgentTorch, DoubleDQNAgent):
    def __init__(self):
        SARSAAgent.__init__(self)
        SARSALamdaAgent.__init__(self)
        DQNAgentTorch.__init__(self)
        DoubleDQNAgent.__init__(self)
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
        logger.info(f"*****启动: {show_policy}*****")
        method_name = "default"
        is_train = False
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
            if show_policy == "深度Q学习算法_pytorch":
                # logger.info(f"启动：深度Q学习算法_pytorch算法")
                if game_round > 0 and game_round % self.update_lr_steps == 0:
                    self.learning_rate *= 0.1
                    logger.info(f"更新学习率:: {self.learning_rate},下降0.1")
                episode_reward = self.play_game_by_dqn_torch_learning(train=False)  # 第round轮次的累积reward
                method_name = self.play_game_by_dqn_torch_learning.__name__
            if show_policy == "Double深度Q学习算法_pytorch":
                # logger.info(f"启动：深度Q学习算法_pytorch算法")
                if game_round > 0 and game_round % self.update_lr_steps == 0:
                    self.ddqn_learning_rate *= 0.1
                    logger.info(f"更新学习率:: {self.ddqn_learning_rate},下降0.1")
                episode_reward = self.play_game_by_double_dqn_torch_learning(train=False)  # 第round轮次的累积reward
                method_name = self.play_game_by_double_dqn_torch_learning.__name__

            if is_train and self.save_policy and (game_round % 150 == 0 or game_round == self.game_rounds - 1):
                if show_policy == "函数近似SARSA算法" or show_policy == "函数近似SARSA(𝜆)算法":
                    save_data = {
                        "weights": self.weights,
                        "encoder": self.tile_coder.codebook if self.tile_coder else None
                    }
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)
                if show_policy == "深度Q学习算法_pytorch":
                    save_data = {"evaluate_net_pytorch": self.evaluate_net_pytorch,
                                 "target_net_pytorch": self.target_net_pytorch,
                                 "optimizer": self.dqn_optimizer}

                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)
                if show_policy == "Double深度Q学习算法_pytorch":
                    save_data = {"ddqn_evaluate_net_pytorch": self.ddqn_evaluate_net_pytorch,
                                 "ddqn_target_net_pytorch": self.ddqn_target_net_pytorch,
                                 "ddqn_optimizer": self.ddqn_optimizer}

                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)

            if episode_reward is not None:
                episode_rewards.append(episode_reward)
                if is_train:
                    if self.learn_step_counter % 10 == 0:  # 每 10 轮记录一次奖励
                        self.writer.add_scalar("Episode Reward", episode_reward, global_step=self.learn_step_counter)
                        self.ddqn_writer.add_scalar("Episode Reward", episode_reward,
                                                    global_step=self.ddqn_learn_step_counter)
                rate_every_length = (round((self.done_rate.count(True) / len(self.done_rate)), 2) * 100)
                logger.info(f"｜第{game_round}轮奖励: ${episode_reward}"
                            f"｜>>>>>>>"
                            f"｜前{len(self.done_rate)}回合成功率:{rate_every_length}%｜")
                if len(self.done_rate) == 100 and rate_every_length >= 80 and is_train:
                    logger.info(f"!!!成功率已经达到70%，自动停止训练!!!")
                    break


            else:
                logger.warning(f"第{game_round}轮奖励为 None，已跳过。")

            Visualizer.plot_cumulative_avg_rewards(episode_rewards, game_round, self.game_rounds, self.class_name,
                                                   method_name)

        print(
            f"平均奖励：{(np.round(np.mean(episode_rewards), 2))} = {np.sum(episode_rewards)} / {len(episode_rewards)}")
        print(
            f"最后100轮奖励：{(np.round(np.mean(episode_rewards[-500:]), 2))} = {np.sum(episode_rewards[-500:])} / {len(episode_rewards[-500:])}")
        logger.info(f"*****结束: {show_policy}*****")
