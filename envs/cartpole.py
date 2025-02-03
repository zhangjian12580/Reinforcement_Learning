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
from tqdm import tqdm
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
        self.game_rounds = 10000
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

    @property
    def print_env_info(self):
        return self.__env_info

    def __env_info(self):
        logger.info(f'观测空间：{self.envs.observation_space}')
        logger.info(f'动作空间：{self.envs.action_space}')
        logger.info(f'位置范围：{(self.envs.min_position, self.envs.max_position)}')
        logger.info(f'速度范围：{(-self.envs.max_speed, self.envs.max_speed)}')
        logger.info(f'目标位置：{self.envs.goal_position}')


class BuildNetwork(nn.Module):
    def __init__(self, hidden_sizes, output_size, activation=nn.ReLU, output_activation=None):
        super(BuildNetwork, self).__init__()

        layers = []

        # 构建隐藏层
        for i, hidden_size in enumerate(hidden_sizes):
            # 对于第一层，我们需要根据输入维度来设置 in_features
            in_features = 4 if i == 0 else hidden_sizes[i - 1]  # 假设输入是 4 维
            layers.append(nn.Linear(in_features=in_features, out_features=hidden_size))
            layers.append(activation())  # 使用指定的激活函数

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
        # 确保 observation 是 PyTorch tensor，并且添加 batch 维度
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # 形状变为 (1, observation_size)

        # 获取策略网络的输出（logits）
        probs = self.policy_net(observation)  # 输出的大小为 (1, Action_Num)

        # 将概率值转换为 NumPy 数组，并进行随机选择动作
        probs = probs.squeeze(0).detach().cpu().numpy()  # .squeeze(0) 去掉 batch 维度，转换为 (Action_Num,)

        # 根据概率分布选择动作
        action = np.random.choice(self.Action_Num, p=probs)
        return action

    def vpg_learn(self, observation, action, reward, done):
        self.trajectory.append((observation, action, reward))

        if done:
            # 将轨迹转换为 Pandas DataFrame
            df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'reward'])
            df['discount'] = self.gamma ** df.index.to_series()
            df['discounted_reward'] = df['discount'] * df['reward']
            df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
            df['psi'] = df['discounted_return']

            # 将输入转换为 Tensor
            x = torch.tensor(np.stack(df['observation']), dtype=torch.float32)

            # 如果有基线网络
            if hasattr(self, 'baseline_net'):
                baseline_output = self.baseline_net(x)  # 输出一个基线值
                df['baseline'] = baseline_output.detach().numpy()  # detach() 不进行梯度计算
                df['psi'] -= (df['baseline'].squeeze() * df['discount'])
                df['return'] = df['discounted_return'] / df['discount']
                y = torch.tensor(df['return'].values[:, np.newaxis], dtype=torch.float32)

                # 基线网络的训练
                self.baseline_optimizer.zero_grad()
                baseline_pred = self.baseline_net(x)
                baseline_loss = nn.MSELoss()(baseline_pred, y)
                baseline_loss.backward()
                self.baseline_optimizer.step()

            # 策略网络训练
            y = torch.tensor(df['psi'].values, dtype=torch.float32)

            self.policy_optimizer.zero_grad()

            # 计算策略网络的输出
            policy_output = self.policy_net(x)

            # 使用负对数似然损失
            log_probs = torch.log(policy_output)
            selected_log_probs = log_probs.gather(1, torch.tensor(df['action'].values, dtype=torch.long).view(-1, 1))
            policy_loss = -(selected_log_probs * y.view(-1, 1)).mean()

            policy_loss.backward()
            self.policy_optimizer.step()

            # 清空轨迹
            self.trajectory = []

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

            # if not train:
            #     logger.info(f"****启动评估阶段****")
            #     self.evaluate_net_pytorch.eval()
            #     self.target_net_pytorch.eval()

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
                    # if train:
                    #     if self.learn_step_counter % 2000 == 0:
                    #         self.epsilon = max(0.01, self.epsilon * 0.995)
                    #     if self.learn_step_counter and self.learn_step_counter % 100 == 0:
                    #         self.target_net_pytorch.load_state_dict(self.evaluate_net_pytorch.state_dict())
                    break

                observation = next_observation
            return episode_reward

class CartPole(VPGAgent):
    def __init__(self):
        VPGAgent.__init__(self)
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

            if self.global_is_train and self.save_policy and (game_round % 150 == 0 or game_round == self.game_rounds - 1):
                if show_policy == "同策策略梯度算法":
                    save_data = {"policy_net": self.policy_net,
                                 "baseline_net": self.baseline_net,
                                 "policy_optimizer": self.policy_optimizer,
                                 "baseline_optimizer": self.baseline_optimizer}
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)

            if episode_reward is not None:
                episode_rewards.append(episode_reward)

                if self.is_open_writer:
                    if self.learn_step_counter % 10 == 0:  # 每 10 轮记录一次奖励
                        self.writer.add_scalar("Episode Reward", episode_reward, global_step=self.learn_step_counter)

                if self.global_is_train:
                    if False not in self.done_rate and np.round(np.mean(episode_rewards[-100:]), 2) >= 197 and self.global_is_train:
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