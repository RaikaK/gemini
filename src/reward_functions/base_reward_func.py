import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import abc

from ygo.udi_io import UdiIO
from ygo.models.duel_state_data import DuelStateData
from ygo.models.command_request import CommandRequest

from src.ygo_env_wrapper.action_data import ActionData


class BaseRewardFunction(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def eval(
        self,
        action_data: ActionData,
    ) -> float:
        """action_dataに含まれるある状態sにおける行動aを評価できる。次状態を考慮した報酬関数を設計する場合、self.udi_ioから、実行後の状態(つまり、次状態)を取得し評価を行う"""
        pass
