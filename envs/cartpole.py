# -*- coding: utf-8 -*-
"""
@File    : cartpole.py
@Time    : 2025/2/1 14:00
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""
import time
from collections import deque

import torch
import gym
import numpy as np
import logging
import torch.nn as nn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from envs.env_template import Env
from tools.visualizer import Visualizer
from tools.save_policy import Policy_loader
import torch.optim as optim

logger = logging.getLogger(__name__)  # 使用当前模块名
from envs.global_set import *


class EnvInit(Env):
    """
    算法参数初始化
    """

    def __init__(self, name='CartPole-v0', render_mode=render_model[0], render=True):
        super().__init__(name, render_mode, render)
        # 是否开启动画
        if render:
            self.env = gym.make(name, render_mode=render_mode)
        else:
            self.env = gym.make(name)

        self.render = render
        # 游戏轮数
        self.game_rounds = 20000
        # 获取动作空间的大小，即可选择的动作数量
        self.Action_Num = self.env.action_space.n
        # 位置
        self.positions = []
        # 用于跟踪最近游戏的完成率
        self.done_rate = deque(maxlen=100)
        self.done_rate.clear()
        # 速度
        self.velocities = []
        # 保存模型
        self.save_policy = False
        # 加载模型
        self.load_model = True
        # 是否开启tensorboard记录logs
        self.is_open_writer = True
        # 是否全局训练，用于设置某些记录
        self.global_is_train = False
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


class BuildNetwork(nn.Module):
    def __init__(self, hidden_sizes, output_size, activation=nn.ReLU, output_activation=None):
        super(BuildNetwork, self).__init__()
        in_features = 4  # 输入状态空间维度
        layers = []

        # 构建隐藏层
        for i, hidden_size in enumerate(hidden_sizes):
            # 对于第一层，我们需要根据输入维度来设置 in_features
            layers.append(nn.Linear(in_features=in_features, out_features=hidden_size))
            layers.append(activation())  # 使用指定的激活函数
            in_features = hidden_size

        # 构建输出层
        layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=output_size))
        if output_activation:
            layers.append(output_activation())  # 如果有输出激活函数

        # 将层组合成一个网络
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_optimizer(self, learning_rate):
        return optim.Adam(self.parameters(), lr=learning_rate)


class VPGAgent(EnvInit):
    def __init__(self, gamma=0.99, learning_rate=0.001):
        """
        同策策略梯度
        :param gamma:
        :param learning_rate:
        """
        super().__init__()
        self.gamma = gamma
        # 其他超参数
        self.learn_step_counter = int(0)  # 学习步计数器
        self.learning_rate = 0.0001  # 学习率
        self.goal_position = 0.5
        self.replay_start_size = 1000  # 经验池开始训练所需的最小样本数量
        self.update_lr_steps = 5000  # 学习率刷新间隔
        current_time = time.localtime()
        log_dir = time.strftime("runs/vpg_agent/%Y_%m_%d_%H_%M", current_time)
        if self.is_open_writer:
            self.writer = SummaryWriter(log_dir=log_dir)
        self.trajectory = []
        policy_kwargs = {
            'hidden_sizes': [10, ],  # 隐藏层大小
            'output_size': 10,  # 输出类别数量
        }
        baseline_kwargs = {
            'hidden_sizes': [10, ],  # 基线网络的隐藏层
        }
        # 构建策略网络
        self.policy_net = BuildNetwork(
            hidden_sizes=policy_kwargs['hidden_sizes'],
            output_size=self.Action_Num,
            activation=policy_kwargs.get('activation', nn.ReLU),
            output_activation=policy_kwargs.get('output_activation', nn.Softmax)
        )
        self.policy_optimizer = self.policy_net.get_optimizer(learning_rate)

        if baseline_kwargs:
            # 构建基线网络
            self.baseline_net = BuildNetwork(
                hidden_sizes=baseline_kwargs['hidden_sizes'],
                output_size=1,  # 基线网络输出一个值
                activation=baseline_kwargs.get('activation', nn.ReLU)
            )
            self.baseline_optimizer = self.baseline_net.get_optimizer(learning_rate)

        # 如果加载模型
        if self.load_model:
            checkpoint = torch.load("tools/policy_dir/CartPole/policy_net.pth", weights_only=True)
            self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->policy_net")
            checkpoint = torch.load("tools/policy_dir/CartPole/baseline_net.pth", weights_only=True)
            self.baseline_net.load_state_dict(checkpoint["model_state_dict"])
            self.baseline_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->baseline_net")

    def vpg_decide(self, observation):
        """
        决策
        :param observation:
        :return:
        """
        # 确保 observation 是 PyTorch tensor，并且添加 batch 维度
        # unsqueeze(0) 将会在第一维添加一个新的维度，形状变为 (1, 4)
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # 形状变为 (1, observation_size)
        # 获取策略网络的输出（logits）
        probs = self.policy_net(observation)  # 输出的大小为 (1, Action_Num)
        # 将概率值转换为 NumPy 数组，并进行随机选择动作
        """
        1. detach() 的作用是从计算图中分离出 probs，即它不再参与后续的梯度计算。
        2. cpu() 将 probs 张量从当前设备（比如 GPU）移动到 CPU 上，
        这对于后续的 numpy() 转换是必需的，因为 numpy() 不支持直接操作 GPU 上的张量
        3. 最后，numpy() 将 PyTorch 的张量转换为 NumPy 数组，NumPy 是 Python 中常用的数组库，
        不支持直接与 PyTorch 张量进行计算，所以需要转换为 NumPy 数组。
        """
        probs = probs.squeeze(0).detach().cpu().numpy()  # .squeeze(0) 去掉 batch 维度，转换为 (Action_Num,)
        # 根据概率分布选择动作
        action = np.random.choice(self.Action_Num, p=probs)

        return action

    def vpg_learn(self, observation, action, reward, done):
        self.trajectory.append((observation, action, reward))

        if done:
            # 将轨迹转换为 Pandas DataFrame
            df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'reward'])
            # 会将索引转换为一个 Series 对象，其中每个元素表示轨迹的一个时间步
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            # 将折扣奖励序列反转，表示从终止状态到开始状态的顺序。强化学习中，通常从终止状态反向计算回报。
            # .cumsum() 是 Pandas 中计算累积和的函数。在这里，它用于计算从反向顺序的折扣奖励序列的累积和。
            # 也就是说，每一步的折扣累积回报（discounted_return）是从后面的奖励开始加权累加的。
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_return']

            # 将输入转换为 Tensor（200，4）
            state = torch.tensor(np.stack(df['observation']), dtype=torch.float32)

            # 如果有基线网络
            # 检查当前对象（self）是否包含 baseline_net 属性
            if hasattr(self, 'baseline_net'):
                baseline_output = self.baseline_net(state)  # 输出一个基线值（200，1）
                # 每个状态的值函数估计
                df['baseline'] = baseline_output.detach().numpy()  # detach() 不进行梯度计算
                # 优势函数
                df['psi'] -= (df['baseline'].squeeze() * df['discount'])
                # 这里计算 df['return'] 列，它通常表示 标准化的回报，使用 除以折扣因子，这是为了消除折扣因子的影响并使回报恢复到接近于“未经折扣的回报”
                df['return'] = df['discounted_return'] / df['discount']
                # df['return'].values 获取 return 列的数据。G
                G = torch.tensor(df['return'].values, dtype=torch.float32).unsqueeze(1)

                # 基线网络的训练
                self.baseline_optimizer.zero_grad()
                V_s = self.baseline_net(state)  # 状态价值估计:v(S;w)
                baseline_loss = nn.MSELoss()(V_s, G)  # V_s:预测(状态估计)， G:实际标准回报
                baseline_loss.backward()
                self.baseline_optimizer.step()

            # 策略网络训练，df['psi'].values：psi 列的numpy数据
            y = torch.tensor(df['psi'].values, dtype=torch.float32)

            self.policy_optimizer.zero_grad()

            # 计算策略网络的输出
            policy_output = self.policy_net(state)

            # 使用负对数似然损失,由于策略梯度方法通常会使用 对数概率 来避免概率值非常小时的数值稳定性问题，
            # 因此这里对 policy_output（动作概率分布）取对数
            log_probs = torch.log(policy_output)
            # gather: 它从给定维度 dim 上根据指定的 index 选择对应的值
            # gather(1, ...) 表示我们从 log_probs 中按列选择特定的动作概率,df['action'].values 表示动作的索引
            # view(-1, 1)：行自动推断，列为一列
            selected_log_probs = log_probs.gather(1, torch.tensor(df['action'].values, dtype=torch.long).view(-1, 1))
            # 最小化负期望回报
            policy_loss = -(selected_log_probs * y.view(-1, 1)).mean()

            policy_loss.backward()
            self.policy_optimizer.step()

            # 清空轨迹
            self.trajectory = []

    def play_montecarlo(self, train=False):
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
            self.policy_net.eval()
            self.baseline_net.eval()

        while True:
            if self.render:
                self.env.render()

            if not train:
                with torch.no_grad():
                    action = self.vpg_decide(observation)
            else:
                action = self.vpg_decide(observation)

            next_observation, reward, terminated, truncated, _ = self.step(action)
            episode_reward += reward

            if terminated or truncated:
                done = True
                # self.learn_step_counter += 1

            if train:
                self.vpg_learn(observation, action, reward, done)
            if done:
                logger.info(f"结束一轮游戏, 奖励为${episode_reward}")
                flag = True if episode_reward >= 195 else False
                self.done_rate.append(flag)
                break

            observation = next_observation
        return episode_reward


class OffPolicyVPGAgent(EnvInit):
    def __init__(self, gamma=0.99, learning_rate=0.001):
        """
        异策策略梯度
        :param gamma:
        :param learning_rate:
        """
        super().__init__()
        self.gamma = gamma
        # 其他超参数
        self.learn_step_counter = int(0)  # 学习步计数器
        self.learning_rate = 0.00001  # 学习率
        self.goal_position = 0.5
        self.replay_start_size = 1000  # 经验池开始训练所需的最小样本数量
        self.update_lr_steps = 5000  # 学习率刷新间隔

        current_time = time.localtime()
        log_dir = time.strftime("runs/off_vpg_agent/%Y_%m_%d_%H_%M", current_time)
        if self.is_open_writer:
            self.writer = SummaryWriter(log_dir=log_dir)

        self.trajectory = []

        def dot(y_true, y_pred):
            return -torch.sum(y_true * y_pred, dim=-1)

        policy_kwargs = {
            'hidden_sizes': [10, ],  # 隐藏层大小
            'output_size': 10,  # 输出类别数量
        }
        baseline_kwargs = {
            'hidden_sizes': [10, ],  # 基线网络的隐藏层
        }
        # 构建策略网络
        self.off_policy_net = BuildNetwork(
            hidden_sizes=policy_kwargs['hidden_sizes'],
            output_size=self.Action_Num,
            activation=policy_kwargs.get('activation', nn.ReLU),
            output_activation=policy_kwargs.get('output_activation', nn.Softmax)
        )
        self.off_policy_optimizer = self.off_policy_net.get_optimizer(learning_rate)

        if baseline_kwargs:
            # 构建基线网络
            self.off_baseline_net = BuildNetwork(
                hidden_sizes=baseline_kwargs['hidden_sizes'],
                output_size=1,  # 基线网络输出一个值
                activation=baseline_kwargs.get('activation', nn.ReLU)
            )
            self.off_baseline_optimizer = self.off_baseline_net.get_optimizer(learning_rate)

        # 如果加载模型
        if self.load_model:
            checkpoint = torch.load("tools/policy_dir/CartPole/off_policy_net.pth", weights_only=True)
            self.off_policy_net.load_state_dict(checkpoint["model_state_dict"])
            self.off_policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->off_policy_net")
            checkpoint = torch.load("tools/policy_dir/CartPole/off_baseline_net.pth", weights_only=True)
            self.off_baseline_net.load_state_dict(checkpoint["model_state_dict"])
            self.off_baseline_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->off_baseline_net")

    def off_vpg_learn(self, observation, action, behavior, reward, done):
        self.trajectory.append((observation, action, behavior, reward))

        if done:
            # 将轨迹转换为 Pandas DataFrame
            df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'behavior', 'reward'])
            # 会将索引转换为一个 Series 对象，其中每个元素表示轨迹的一个时间步
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            # 将折扣奖励序列反转，表示从终止状态到开始状态的顺序。强化学习中，通常从终止状态反向计算回报。
            # .cumsum() 是 Pandas 中计算累积和的函数。在这里，它用于计算从反向顺序的折扣奖励序列的累积和。
            # 也就是说，每一步的折扣累积回报（discounted_return）是从后面的奖励开始加权累加的。
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_return']

            # 将输入转换为 Tensor（200，4）
            state = torch.tensor(np.stack(df['observation']), dtype=torch.float32)

            # 如果有基线网络
            # 检查当前对象（self）是否包含 baseline_net 属性
            if hasattr(self, 'baseline_net'):
                baseline_output = self.baseline_net(state)  # 输出一个基线值（200，1）
                # 每个状态的值函数估计
                df['baseline'] = baseline_output.detach().numpy()  # detach() 不进行梯度计算
                # 优势函数
                df['psi'] -= (df['baseline'].squeeze() * df['discount'])
                # 这里计算 df['return'] 列，它通常表示 标准化的回报，使用 除以折扣因子，这是为了消除折扣因子的影响并使回报恢复到接近于“未经折扣的回报”
                df['return'] = df['discounted_return'] / df['discount']
                # df['return'].values 获取 return 列的数据。G
                G = torch.tensor(df['return'].values, dtype=torch.float32).unsqueeze(1)

                # 基线网络的训练
                self.off_baseline_optimizer.zero_grad()
                V_s = self.baseline_net(state)  # 状态价值估计:v(S;w)
                baseline_loss = nn.MSELoss()(V_s, G)  # V_s:预测(状态估计)， G:实际标准回报
                baseline_loss.backward()
                self.off_baseline_optimizer.step()

            # 策略网络训练，df['psi'].values：psi 列的numpy数据
            y = torch.tensor((df['psi'] / df['behavior']).values, dtype=torch.float32)

            self.off_policy_optimizer.zero_grad()

            # 计算策略网络的输出
            policy_output = self.off_policy_net(state)

            # 使用负对数似然损失,由于策略梯度方法通常会使用 对数概率 来避免概率值非常小时的数值稳定性问题，
            # 因此这里对 policy_output（动作概率分布）取对数
            log_probs = torch.log(policy_output)
            # gather: 它从给定维度 dim 上根据指定的 index 选择对应的值
            # gather(1, ...) 表示我们从 log_probs 中按列选择特定的动作概率,df['action'].values 表示动作的索引
            # view(-1, 1)：行自动推断，列为一列

            selected_log_probs = log_probs.gather(1, torch.tensor(df['action'].values, dtype=torch.long).view(-1, 1))

            # 最小化负期望回报
            policy_loss = -(selected_log_probs * y.view(-1, 1)).mean()

            policy_loss.backward()
            self.off_policy_optimizer.step()

            # 清空轨迹
            self.trajectory = []

    def off_decide(self, observation):
        # 确保 observation 是 PyTorch tensor，并且添加 batch 维度
        # unsqueeze(0) 将会在第一维添加一个新的维度，形状变为 (1, 4)
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # 形状变为 (1, observation_size)
        # 获取策略网络的输出（logits）
        probs = self.off_policy_net(observation)  # 输出的大小为 (1, Action_Num)
        # 将概率值转换为 NumPy 数组，并进行随机选择动作
        """
        1. detach() 的作用是从计算图中分离出 probs，即它不再参与后续的梯度计算。
        2. cpu() 将 probs 张量从当前设备（比如 GPU）移动到 CPU 上，
        这对于后续的 numpy() 转换是必需的，因为 numpy() 不支持直接操作 GPU 上的张量
        3. 最后，numpy() 将 PyTorch 的张量转换为 NumPy 数组，NumPy 是 Python 中常用的数组库，
        不支持直接与 PyTorch 张量进行计算，所以需要转换为 NumPy 数组。
        """
        probs = probs.squeeze(0).detach().cpu().numpy()  # .squeeze(0) 去掉 batch 维度，转换为 (Action_Num,)
        # 根据概率分布选择动作
        action = np.random.choice(self.Action_Num, p=probs)
        behavior = 1. / self.Action_Num
        return action, behavior

    def off_play_montecarlo(self, train=False):
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
            self.off_policy_net.eval()
            self.off_baseline_net.eval()

        while True:
            if self.render:
                self.env.render()

            if not train:
                with torch.no_grad():
                    action, behavior = self.off_decide(observation)
            else:
                action, behavior = self.off_decide(observation)

            next_observation, reward, terminated, truncated, _ = self.step(action)
            episode_reward += reward

            if terminated or truncated:
                done = True
                # self.learn_step_counter += 1

            if train:
                self.off_vpg_learn(observation, action, behavior, reward, done)
            if done:
                logger.info(f"结束一轮游戏, 奖励为${episode_reward}")
                flag = True if episode_reward >= 195 else False
                self.done_rate.append(flag)
                break

            observation = next_observation
        return episode_reward


class CartPole(VPGAgent, OffPolicyVPGAgent):
    def __init__(self):
        VPGAgent.__init__(self)
        OffPolicyVPGAgent.__init__(self)
        self.class_name = self.__class__.__name__

    def game_iteration(self, show_policy):
        """
        迭代
        :param show_policy: 使用的更新策略方式
        """
        episode_reward = 0.
        episode_rewards = []  # 总轮数的奖励(某轮总奖励)列表
        logger.info(f"*****启动: {show_policy}*****")
        method_name = "default"
        for game_round in range(1, self.game_rounds):
            logger.info(f"---第{game_round}轮训练---")

            if show_policy == "同策策略梯度算法":
                # logger.info(f"函数近似SARSA算法")
                episode_reward = self.play_montecarlo(train=True)  # 第round轮次的累积reward
                method_name = self.play_montecarlo.__name__

            if show_policy == "异策策略梯度算法":
                # logger.info(f"函数近似SARSA算法")
                episode_reward = self.off_play_montecarlo(train=False)  # 第round轮次的累积reward
                method_name = self.off_play_montecarlo.__name__

            if self.global_is_train and self.save_policy and (
                    game_round % 150 == 0 or game_round == self.game_rounds - 1):
                if show_policy == "同策策略梯度算法":
                    save_data = {"policy_net": self.policy_net,
                                 "baseline_net": self.baseline_net,
                                 "policy_optimizer": self.policy_optimizer,
                                 "baseline_optimizer": self.baseline_optimizer}
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)
                if show_policy == "异策策略梯度算法":
                    save_data = {"off_policy_net": self.off_policy_net,
                                 "off_baseline_net": self.off_baseline_net,
                                 "off_policy_optimizer": self.off_policy_optimizer,
                                 "off_baseline_optimizer": self.off_baseline_optimizer}
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)

            if episode_reward is not None:
                episode_rewards.append(episode_reward)
                if self.is_open_writer:
                    if self.learn_step_counter % 10 == 0:  # 每 10 轮记录一次奖励
                        self.writer.add_scalar("Episode Reward", episode_reward, global_step=self.learn_step_counter)
                if self.global_is_train:
                    if False not in self.done_rate and np.round(np.mean(episode_rewards[-100:]),
                                                                2) >= 197 and self.global_is_train:
                        logger.info(f"!!!成功率已经达到百回合195，自动停止训练!!!")
                        break
            else:
                logger.warning(f"第{game_round}轮奖励为 None，已跳过。")

            Visualizer.plot_cumulative_avg_rewards(episode_rewards, game_round, self.game_rounds, self.class_name,
                                                   method_name)

        print(
            f"平均奖励：{(np.round(np.mean(episode_rewards), 2))} = {np.sum(episode_rewards)} / {len(episode_rewards)}")
        print(
            f"最后100轮奖励：{(np.round(np.mean(episode_rewards[-100:]), 2))} = {np.sum(episode_rewards[-100:])} / {len(episode_rewards[-100:])}")
        logger.info(f"*****结束: {show_policy}*****")
