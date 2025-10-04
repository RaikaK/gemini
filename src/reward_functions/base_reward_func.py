import sys

sys.path.append("C:/Users/b1/Desktop/u-ni-yo")

import abc

from ygo.models.duel_state_data import DuelStateData
from ygo.models.duel_log_data import DuelEndData

from src.ygo_env_wrapper.action_data import ActionData


class BaseRewardFunction(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def eval(
        self, action_data: ActionData, duel_state_data: DuelStateData, is_duel_end: bool, duel_end_data: DuelEndData
    ) -> float:
        """action_dataの結果、ゲームの状態がどのように変化したかを評価する"""
        pass
