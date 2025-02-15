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
import torch.nn.functional as F

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
        self.done_rate = deque(maxlen=300)
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
        if output_activation is not None:
            # 如果是 Softmax，需要指定维度
            if output_activation == nn.Softmax:
                layers.append(output_activation(dim=-1))
            else:
                layers.append(output_activation())

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
        # 因为是平衡游戏，所以只有当游戏强行停止时(也就是达到200步骤)，才会结束
        if done:
            # 将轨迹转换为 Pandas DataFrame
            df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'reward'])
            # df.index.to_series()会将索引转换为一个 Series 对象，=0,1,2,...,200，将索引作为幂次方对象
            df['discount'] = self.gamma ** df.index.to_series()
            # df['discounted_reward']相当于每个奖励都会带一个折扣因子
            df['discounted_reward'] = df['discount'] * df['reward']
            # 将折扣奖励序列反转，表示从终止状态到开始状态的顺序。强化学习中，通常从终止状态反向计算回报。
            # .cumsum() 是 Pandas 中计算累积和的函数。在这里，它用于计算从反向顺序的折扣奖励序列的累积和。
            # 也就是说，每一步的折扣累积回报（discounted_return）是从后面的奖励开始加权累加的。
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()  # 第一项就存在折扣因子
            df['psi'] = df['discounted_return']

            # 将输入转换为 Tensor（200，4）
            state = torch.tensor(np.stack(df['observation']), dtype=torch.float32)

            # 如果有基线网络
            # 检查当前对象（self）是否包含 baseline_net 属性
            if hasattr(self, 'baseline_net'):
                baseline_output = self.baseline_net(state)  # 输出一个基线值（200，1）
                # 每个状态的值函数估计
                df['baseline'] = baseline_output.detach().numpy()  # detach() 不进行梯度计算
                # 优势函数，因为基线网络的输出是状态价值，第一项没有折扣因子，为了与df['psi']匹配需要乘上
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

            # 策略网络训练，df['psi'].values：psi 列的numpy数据，无论是否归一化，代表的是一种相对影响，只要体现正负关系即可。
            advantage = torch.tensor(df['psi'].values, dtype=torch.float32)

            self.policy_optimizer.zero_grad()

            # 计算策略网络的输出，𝛑(a|s)
            policy_output = self.policy_net(state)

            # 使用负对数似然损失,由于策略梯度方法通常会使用 对数概率 来避免概率值非常小时的数值稳定性问题，
            # 因此这里对 policy_output（动作概率分布）取对数
            log_probs = torch.log(policy_output)
            # gather: 它从给定维度 dim 上根据指定的 index 选择对应的值
            # gather(1, ...) 表示我们从 log_probs 中按列选择特定的动作概率,df['action'].values 表示action中的值-动作的索引
            # view(-1, 1)：（-1）行自动推断，列为1列
            selected_log_probs = log_probs.gather(1, torch.tensor(df['action'].values, dtype=torch.long).view(-1, 1))
            # 最小化负期望回报
            policy_loss = -(selected_log_probs * advantage.view(-1, 1)).mean()

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
        :param gamma: 折扣因子
        :param learning_rate: 学习率
        """
        super().__init__()
        self.gamma = gamma

        # 超参数设置
        self.learn_step_counter = int(0)
        self.learning_rate = 0.001  # 调整学习率
        self.goal_position = 0.5
        self.replay_start_size = 1000
        self.update_lr_steps = 5000
        self.temperature = 1.0  # 添加温度参数
        self.min_temperature = 0.01
        self.temperature_decay = 0.995

        # TensorBoard 设置
        current_time = time.localtime()
        log_dir = time.strftime("runs/off_vpg_agent/%Y_%m_%d_%H_%M", current_time)
        if self.is_open_writer:
            self.writer = SummaryWriter(log_dir=log_dir)

        self.trajectory = []

        # 网络配置
        policy_kwargs = {
            'hidden_sizes': [64, 64],
            'activation': nn.ReLU,
            'output_activation': lambda x: nn.Softmax(dim=-1)(x)  # 显式指定 Softmax 维度
        }

        baseline_kwargs = {
            'hidden_sizes': [64, 64],
            'activation': nn.ReLU,
            'output_activation': None
        }

        # 构建策略网络
        self.off_policy_net = BuildNetwork(
            hidden_sizes=policy_kwargs['hidden_sizes'],
            output_size=self.Action_Num,
            activation=nn.ReLU,
            output_activation=nn.Softmax  # 直接传入 Softmax 类
        )

        # 构建基线网络
        self.off_baseline_net = BuildNetwork(
            hidden_sizes=baseline_kwargs['hidden_sizes'],
            output_size=1,
            activation=baseline_kwargs['activation'],
            output_activation=baseline_kwargs['output_activation']
        )

        # 优化器
        self.off_policy_optimizer = self.off_policy_net.get_optimizer(learning_rate)
        self.off_baseline_optimizer = self.off_baseline_net.get_optimizer(learning_rate)

        # 学习率调度器
        self.policy_scheduler = optim.lr_scheduler.StepLR(
            self.off_policy_optimizer, step_size=1000, gamma=0.9
        )
        self.baseline_scheduler = optim.lr_scheduler.StepLR(
            self.off_baseline_optimizer, step_size=1000, gamma=0.9
        )

        # 加载模型
        if self.load_model:
            self._load_models()

    def _load_models(self):
        """加载预训练模型"""
        try:
            policy_checkpoint = torch.load("tools/policy_dir/CartPole/off_policy_net.pth")
            self.off_policy_net.load_state_dict(policy_checkpoint["model_state_dict"])
            self.off_policy_optimizer.load_state_dict(policy_checkpoint["optimizer_state_dict"])

            baseline_checkpoint = torch.load("tools/policy_dir/CartPole/off_baseline_net.pth")
            self.off_baseline_net.load_state_dict(baseline_checkpoint["model_state_dict"])
            self.off_baseline_optimizer.load_state_dict(baseline_checkpoint["optimizer_state_dict"])

            logger.info("成功加载异策略网络模型")
        except Exception as e:
            logger.warning(f"加载模型失败: {e}")

    def off_decide(self, observation):
        """
        根据观察选择动作
        :param observation: 环境观察
        :return: action, behavior_prob
        """
        state = torch.tensor(observation, dtype=torch.float32)
        logits = self.off_policy_net(state)
        probs = F.softmax(logits / self.temperature, dim=-1)

        # 使用 epsilon-greedy 策略
        if np.random.rand() < 0.01 and self.global_is_train:  # epsilon = 0.1
            action = np.random.choice(self.Action_Num)
            behavior = 1.0 / self.Action_Num
        else:
            action = np.random.choice(self.Action_Num, p=probs.detach().numpy())
            behavior = probs[action].item()

        return action, behavior

    def off_vpg_learn(self, observation, action, behavior, reward, done):
        """
        异策略VPG学习
        """
        self.trajectory.append((observation, action, behavior, reward))

        if done:
            # 数据预处理
            df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'behavior', 'reward'])

            # 计算折扣回报
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()

            # 归一化奖励
            df['reward'] = (df['reward'] - df['reward'].mean()) / (df['reward'].std() + 1e-8)

            # 转换为张量，np.stack将df转换为numpy，维度不变
            states = torch.tensor(np.stack(df['observation']), dtype=torch.float32)
            actions = torch.tensor(df['action'].values, dtype=torch.long)
            behaviors = torch.tensor(df['behavior'].values, dtype=torch.float32)
            returns = torch.tensor(df['discounted_return'].values, dtype=torch.float32)

            # 计算基线值，状态价值
            baseline_values = self.off_baseline_net(states).squeeze()

            # 计算优势函数
            advantages = returns - baseline_values.detach()

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 计算重要性权重
            current_probs = F.softmax(self.off_policy_net(states), dim=-1)
            # 根据动作选择概率
            selected_probs = current_probs.gather(1, actions.unsqueeze(1)).squeeze()
            # torch.clamp(..., 0.1, 10.0)：对计算出的权重进行裁剪（限制范围）
            importance_weights = torch.clamp(selected_probs / (behaviors + 1e-8), 0.1, 10.0)

            # 训练基线网络
            self.off_baseline_optimizer.zero_grad()
            baseline_loss = F.mse_loss(baseline_values, returns)
            baseline_loss.backward()
            # 模型参数的梯度进行裁剪 的操作，目的是防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.off_baseline_net.parameters(), 1.0)
            self.off_baseline_optimizer.step()

            # 训练策略网络
            self.off_policy_optimizer.zero_grad()
            log_probs = F.log_softmax(self.off_policy_net(states), dim=-1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1))
            policy_loss = -(selected_log_probs * advantages.unsqueeze(1) * importance_weights.unsqueeze(1)).mean()

            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.off_policy_net.parameters(), 1.0)
            self.off_policy_optimizer.step()

            # 更新学习率
            self.policy_scheduler.step()
            self.baseline_scheduler.step()

            # 衰减温度参数
            # 当 T 较大时，策略趋向于更加随机（探索更多可能性）
            # 当 T 较小时，策略趋向于更加确定性（选择高价值的动作）
            self.temperature = max(self.min_temperature,
                                   self.temperature * self.temperature_decay)

            # 记录训练信息
            if self.is_open_writer:
                self.writer.add_scalar('Loss/Policy', policy_loss.item(), self.learn_step_counter)
                self.writer.add_scalar('Loss/Baseline', baseline_loss.item(), self.learn_step_counter)
                self.writer.add_scalar('Values/Temperature', self.temperature, self.learn_step_counter)

            self.learn_step_counter += 1
            self.trajectory = []

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

        while True:
            if self.render:
                self.env.render()

            if not train:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(observation)
                    probs = self.off_policy_net(state_tensor)
                    # 使用确定性策略
                    action = torch.argmax(probs).item()
                    behavior = 1
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
                flag = True if episode_reward >= 197 else False
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
                episode_reward = self.play_montecarlo(train=False)  # 第round轮次的累积reward
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
                    save_data = {
                        "off_policy_net": self.off_policy_net,
                        "off_baseline_net": self.off_baseline_net,
                        "off_policy_optimizer": self.off_policy_optimizer,
                        "off_baseline_optimizer": self.off_baseline_optimizer
                    }
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)

            if episode_reward is not None:
                episode_rewards.append(episode_reward)
                if self.is_open_writer:
                    if self.learn_step_counter % 10 == 0:  # 每 10 轮记录一次奖励
                        self.writer.add_scalar("Episode Reward", episode_reward, global_step=self.learn_step_counter)
                if self.global_is_train:
                    if False not in self.done_rate and np.round(np.mean(episode_rewards[-300:]),
                                                                2) >= 198 and self.global_is_train:
                        logger.info(f"!!!成功率已经达到百回合195，自动停止训练!!!")
                        break
            else:
                logger.warning(f"第{game_round}轮奖励为 None，已跳过。")

            Visualizer.plot_cumulative_avg_rewards(episode_rewards, game_round, self.game_rounds, self.class_name,
                                                   method_name)

        print(
            f"平均奖励：{(np.round(np.mean(episode_rewards), 2))} = {np.sum(episode_rewards)} / {len(episode_rewards)}")
        print(
            f"最后100轮奖励：{(np.round(np.mean(episode_rewards[-300:]), 2))} = {np.sum(episode_rewards[-100:])} / {len(episode_rewards[-100:])}")
        logger.info(f"*****结束: {show_policy}*****")
