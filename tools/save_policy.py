# -*- coding: utf-8 -*-
"""
@File    : save_policy.py      # 文件名，save_policy表示当前文件名
@Time    : 2024/12/31         # 创建时间，2024/12/31表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
import numpy as np
import os

class Policy_loader:
    policy_dir = os.path.join(os.path.dirname(__file__), 'policy_dir')
    if not os.path.exists(policy_dir):
        os.makedirs(policy_dir)
    save_dir = None
    @staticmethod
    def save_policy(method_name, class_name, policy):
        if method_name is None:
            method_name = "default"
        policy_dir = os.path.join(Policy_loader.policy_dir, class_name)
        if not os.path.exists(policy_dir):
            os.makedirs(policy_dir)
        Policy_loader.save_dir = os.path.join(policy_dir, method_name)
        np.savetxt(f'{Policy_loader.save_dir}.csv', policy, delimiter=',', fmt='%.6f')

    @staticmethod
    def load_policy(class_name, method_name):
        # 加载指定路径下的 CSV 文件
        policy_dir = os.path.join(Policy_loader.policy_dir, class_name)
        Policy_loader.save_dir = os.path.join(policy_dir, method_name)
        q_sa_loaded = np.loadtxt(f'{Policy_loader.save_dir}', delimiter=',')

        return q_sa_loaded