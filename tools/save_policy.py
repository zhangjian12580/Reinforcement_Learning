# -*- coding: utf-8 -*-
"""
@File    : save_policy.py      # 文件名，save_policy表示当前文件名
@Time    : 2024/12/31         # 创建时间，2024/12/31表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
import pickle
import torch
import keras
import numpy as np
import os

from envs.blackjack import logger


class Policy_loader:
    policy_dir = os.path.join(os.path.dirname(__file__), 'policy_dir')
    if not os.path.exists(policy_dir):
        os.makedirs(policy_dir)
    save_dir = None
    @staticmethod
    def save_policy(method_name, class_name, policy, **kwargs):
        step = kwargs.get("step", 1)
        if method_name is None:
            method_name = "default"
        policy_dir = os.path.join(Policy_loader.policy_dir, class_name)
        if not os.path.exists(policy_dir):
            os.makedirs(policy_dir)
        Policy_loader.save_dir = os.path.join(policy_dir, method_name)
        if isinstance(policy, dict) and "encoder" in policy:
            with open(f"{Policy_loader.save_dir}.pkl", "wb") as f:
                pickle.dump(policy, f)
        elif isinstance(policy, list):
            np.savetxt(f'{Policy_loader.save_dir}.csv', policy, delimiter=',', fmt='%.6f')
        elif "evaluate_net_pytorch" in policy:
            evaluate_net_dir_py = os.path.join(policy_dir, f'evaluate_net_pytorch')
            target_net_dir_py = os.path.join(policy_dir, f'target_net_pytorch')
            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict':policy['evaluate_net_pytorch'].state_dict(),
                        'optimizer_state_dict': policy['optimizer'].state_dict(),}, f'{evaluate_net_dir_py}.pth')

            torch.save({'model_state_dict':policy['target_net_pytorch'].state_dict(),
                        'optimizer_state_dict': policy['optimizer'].state_dict(),}, f'{target_net_dir_py}.pth')
            logger.info(f"保存-->evaluate_net_pytorch+-->target_net_pytorch模型")
        elif "ddqn_evaluate_net_pytorch" in policy:
            evaluate_net_dir_py = os.path.join(policy_dir, 'ddqn_evaluate_net_pytorch')
            target_net_dir_py = os.path.join(policy_dir, 'ddqn_target_net_pytorch')
            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict': policy['ddqn_evaluate_net_pytorch'].state_dict(),
                        'optimizer_state_dict': policy['ddqn_optimizer'].state_dict(), }, f'{evaluate_net_dir_py}.pth')

            torch.save({'model_state_dict': policy['ddqn_target_net_pytorch'].state_dict(),
                        'optimizer_state_dict': policy['ddqn_optimizer'].state_dict(), }, f'{target_net_dir_py}.pth')
            logger.info(f"保存-->ddqn_evaluate_net_pytorch+-->ddqn_target_net_pytorch模型")
            # torch.save(policy['target_net_pytorch'].state_dict(), f'{target_net_dir_py}.pth')

    @staticmethod
    def load_policy(class_name, method_name):
        # 加载指定路径下的 CSV 文件
        policy_dir = os.path.join(Policy_loader.policy_dir, class_name)
        Policy_loader.save_dir = os.path.join(policy_dir, method_name)

        q_sa_loaded = np.loadtxt(f'{Policy_loader.save_dir}', delimiter=',')

        return q_sa_loaded

    @staticmethod
    def load_w_para(class_name, method_name):
        policy_dir = os.path.join(Policy_loader.policy_dir, class_name)
        Policy_loader.save_dir = os.path.join(policy_dir, method_name)

        with open(f"{Policy_loader.save_dir}", "rb") as f:
            data = pickle.load(f)
        print(f"模型已加载自 {Policy_loader.save_dir}")
        return data["weights"], data["encoder"]

    @staticmethod
    def load_dqn_network(dir):
        loaded_model = keras.models.load_model(filepath=f'{dir}.h5')
        print(f"模型已加载自 {Policy_loader.save_dir}")
        return loaded_model