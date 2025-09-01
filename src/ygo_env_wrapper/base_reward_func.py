import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import abc

from ygo.udi_io import UdiIO
from ygo.models.duel_state_data import DuelStateData
from ygo.models.command_request import CommandRequest


class BaseRewardFunction(abc.ABC):
    def __init__(self, udi_io: UdiIO, is_normalized: bool):
        self.udi_io = udi_io

        # 報酬の範囲を[-1, 1]にクランプするかどうか
        self.is_normalized = is_normalized

    @abc.abstractmethod
    def eval(
        self,
        duel_state_data: DuelStateData,
        cmd_request: CommandRequest,
        cmd_index: int,
    ) -> float:
        """状態(duel_state_data), 選択可能なコマンド(cmd_request), 実際に実行したコマンド(cmd_index)に基づいて報酬を計算する"""
        pass
