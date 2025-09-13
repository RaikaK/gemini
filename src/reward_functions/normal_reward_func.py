import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

from ygo.models.duel_log_data import DuelEndData

from ygo.constants.enums import ResultType
import ygo.constants as c

from src.reward_functions.base_reward_func import BaseRewardFunction
from src.ygo_env_wrapper.action_data import ActionData


class NormalRewardFunction(BaseRewardFunction):
    def __init__(self):
        super().__init__()

    def eval(self, action_data: ActionData = None) -> float:
        """ゲームに勝利した場合: 1.0, 負けた場合: -1.0, それ以外: 0.0"""
        state: dict = action_data.state
        # 終了かどうかを判定
        is_duel_end = state["is_duel_end"]
        if not is_duel_end:
            return 0.0  # 終了していないので0.0を返す

        duel_end_data: DuelEndData = state["duel_end_data"]
        result_type = duel_end_data.result_type
        if result_type == c.ResultType.WIN:
            return 1.0
        elif result_type == ResultType.LOSE:
            return -1.0

        return 0.0
