import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

from ygo.udi_io import UdiIO
from ygo.models.duel_state_data import DuelStateData
from ygo.models.command_request import CommandRequest, CommandEntry

from reward_functions.base_reward_func import BaseRewardFunction
from src.reward_functions.normal_reward_func import NormalRewardFunction
from src.ygo_env_wrapper.action_data import ActionData


class YgoEnv:
    def __init__(self, udi_io: UdiIO, reward_func: BaseRewardFunction):
        self.udi_io = udi_io  # MDクライアントの情報参照

        self.reward_func = reward_func  # 報酬関数
        # 何も指定がない場合は、 NormalRewardFunctionを使用
        if self.reward_func is None:
            self.reward_func = NormalRewardFunction(udi_io, is_normalized=True)

    def step(
        self, action_data: ActionData
    ) -> dict[str, DuelStateData | bool | CommandRequest | float | bool]:
        """
        コマンド(cmd_index)を実行し、
        - 次状態: "next_state"
        - コマンド選択を要求されているかどうか: "is_cmd_required"
        - コマンドリクエスト: "command_request"
        - 報酬: "reward"
        - 終了フラグ: "done"
        を返す
        """
        cmd_index = action_data.command_index

        # cmd_indexの実行
        self.udi_io.output_command(cmd_index)

        # ##############
        # コマンド実行後
        # ##############
        next_state: DuelStateData = self.udi_io.get_duel_state_data()  # 次状態
        is_cmd_required = (
            self.udi_io.is_command_required()
        )  # コマンド入力が必要かどうか
        command_request = (
            self.udi_io.get_command_request()
        )  # 選択可能なコマンドリストなどの情報

        # 報酬を計算
        reward: float = self.reward_func.eval(action_data=action_data)

        done = self.udi_io.is_duel_end()  # 終了フラグ

        result_dict = {
            "next_state": next_state,
            "is_cmd_required": is_cmd_required,
            "command_request": command_request,
            "reward": reward,
            "done": done,
        }

        return result_dict
