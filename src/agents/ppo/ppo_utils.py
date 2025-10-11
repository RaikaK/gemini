import numpy as np
import torch
from torch.distributions import Categorical

from ygo.models.command_request import CommandEntry, CommandRequest
from ygo.models.duel_state_data import DuelStateData
from src.env.state_data import StateData
from src.env.action_data import ActionData

"""
PPOで使用する汎用関数
"""


def compute_returns_and_advantages(
    state_values: list[float],
    rewards: list[float],
    dones: list[bool],
    gamma: float = 0.9,
    lambda_gae: float = 0.95,
) -> tuple[list[np.float32], list[np.float32]]:
    td_errors = [
        rewards[i] + gamma * state_values[i + 1] - state_values[i]
        if not dones[i]
        else rewards[i] - state_values[i]
        for i in range(len(rewards))
    ]
    # done = Trueの時 → td_error = r - v
    breakpoint()  # len(rewards)とlen(td_errors)を確認 → rewardsの方が1つ多い | donesの中身も確認

    # GAEの計算
    gaes = [None for t in range(len(td_errors))]
    for i in reversed(range(len(td_errors))):
        if not dones[i]:
            gaes[i] = td_errors[i] + gamma * lambda_gae * (1 - dones[i]) * gaes[i + 1]
        else:
            gaes[i] = td_errors[i]

    returns = [gae + state_value for gae, state_value in zip(gaes, state_values)]

    # advantage
    mean_gae = np.mean(gaes)
    std_gae = np.std(gaes) + 1e-8
    advantages = [gae - mean_gae / std_gae for gae in gaes]

    breakpoint()  # gaeの中身を確認 | returnsの長さを確認
    return returns, advantages


def calc_entropy(action_prob: list[float]) -> float:
    """エントロピーを計算する"""
    dist = Categorical(action_prob)
    return dist.entropy()
