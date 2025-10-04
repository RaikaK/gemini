import sys

sys.path.append("C:/Users/b1/Desktop/u-ni-yo")

from ygo.models.duel_state_data import DuelStateData
from ygo.models.duel_log_data import DuelEndData
from ygo.constants.enums import ResultType
import ygo.constants as c

from src.reward_functions.base_reward_func import BaseRewardFunction
from src.ygo_env_wrapper.action_data import ActionData


class NormalRewardFunction(BaseRewardFunction):
    def __init__(self):
        super().__init__()

    def eval(
        self, action_data: ActionData, duel_state_data: DuelStateData, is_duel_end: bool, duel_end_data: DuelEndData
    ) -> float:
        """ゲームに勝利した場合: 1.0, 負けた場合: -1.0, それ以外: 0.0"""
        # 終了かどうかを判定
        if not is_duel_end:
            return 0.0  # 終了していないので0.0を返す

        result_type = duel_end_data.result_type
        if result_type == c.ResultType.WIN:
            return 1.0
        elif result_type == ResultType.LOSE:
            return -1.0

        return 0.0
