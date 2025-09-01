import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

from ygo.udi_io import UdiIO
from ygo.models.duel_state_data import DuelStateData
from ygo.models.command_request import CommandRequest

from src.ygo_env_wrapper.base_reward_func import BaseRewardFunction
from src.reward_functions.normal_reward_func import NormalRewardFunction


class YgoEnv:
    def __init__(self, udi_io: UdiIO, reward_func: BaseRewardFunction):
        self.udi_io = udi_io
        self.reward_func = reward_func
        # 何も指定がない場合は、 NormalRewardFunctionを使用
        if self.reward_func is None:
            self.reward_func = NormalRewardFunction(udi_io, is_normalized=True)

    def reset(self):
        pass

    def step(self, cmd_index: int):
        """CommandRequest.commandsとして受け取ったコマンドリストのインデックスを送信することで、アクションを実行する"""

        # cmd_indexの実行
        self.udi_io.output_command(cmd_index)

        # コマンド入力が求められているまで待機
        is_cmd_required = False
        while not is_cmd_required:
            is_cmd_required = self.udi_io.is_command_required()

        # 状態
        duel_state_data: DuelStateData = self.udi_io.get_duel_state_data()

        # 選択可能なコマンド
        cmd_request: CommandRequest = self.udi_io.get_command_request()

        # 報酬を計算
        reward: float = self.reward_func.eval(duel_state_data, cmd_request, cmd_index)

        return duel_state_data, cmd_request, reward
