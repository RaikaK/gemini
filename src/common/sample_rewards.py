from ygo.models.duel_state_data import DuelStateData
from ygo.models import DuelEndData
from ygo.constants.enums import FinishType

from src.env.state_data import StateData


MAX_LP_POINT = 8000


def life_point_reward(state: StateData, next_state: StateData) -> float:
    """ライフポイントの変化に基づく報酬を計算する関数

    Args:
        state (dict): 現在の状態
        action_data (ActionData): 実行したアクション
        next_state (dict): 次の状態

    Returns:
        float: 計算された報酬
    """
    duel_state_data: DuelStateData = state.duel_state_data
    player_lp = duel_state_data.general_data.lp[0]
    enemy_lp = duel_state_data.general_data.lp[1]

    after_action_player_lp = next_state.duel_state_data.general_data.lp[0]
    after_action_enemy_lp = next_state.duel_state_data.general_data.lp[1]

    player_lp_diff = (after_action_player_lp - player_lp) / MAX_LP_POINT  # 1step前からどの程度LPが減っているか
    enemy_lp_diff = (after_action_enemy_lp - enemy_lp) / MAX_LP_POINT  # 1step前からどの程度LPが減っているか

    reward = player_lp_diff - enemy_lp_diff
    return reward


def nodeck_reward(duel_end_data: DuelEndData | None) -> float:
    if duel_end_data is None:
        return 0
    if duel_end_data.finish_type == FinishType.NO_DECK:
        return -1
    return 0
