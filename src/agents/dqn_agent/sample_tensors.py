import numpy as np

from ygo import constants
from ygo import models

# デッキに使うカードをセット（とりあえず、全部入り）
OneHotTable = {
    1001: 0,
    1002: 1,
    1003: 2,
    1004: 3,
    1005: 4,
    1006: 5,
    1007: 6,
    1008: 7,
    1009: 8,
    1010: 9,
    1011: 10,
    1012: 11,
    1013: 12,
    1014: 13,
    1015: 14,
    1016: 15,
    1017: 16,
    1018: 17,
    1019: 18,
    1020: 19,
    1021: 20,
    1022: 21,
    1023: 22,
    1024: 23,
    1025: 24,
    1026: 25,
    1027: 26,
    1028: 27,
    1029: 28,
    1030: 29,
    1031: 30,
    1032: 31,
    1033: 32,
    1034: 33,
    1035: 34,
}

TABLE_NUM = len(OneHotTable)


# 盤面情報用ベクトル数（SetBoardVector関数で使用）
BOARD_NUM = TABLE_NUM * constants.PosId.UPPER_VALUE
# ゲーム情報用ベクトル数（SetBoardVector関数で使用）
INFO_NUM = 2 + constants.Phase.UPPER_VALUE + (constants.SelectionType.UPPER_VALUE + 1)
# アクション用ベクトル数（SetActionVector関数で使用）
ACTION_NUM = (constants.CommandType.UPPER_VALUE + 1) + (TABLE_NUM + 3)


def set_board_vector(input_data):
    """
    盤面情報をベクトル化する
    """
    duel_table = input_data[2]
    n = len(duel_table)

    v = np.zeros(BOARD_NUM + INFO_NUM, dtype=np.float32)

    # 自分のカードの盤面情報をセット（このサンプルでは相手の情報は見ません）
    for i in range(n):
        d: models.DuelCard = duel_table[i]
        if d.player_id == constants.PlayerId.MYSELF:
            assert d.pos_id >= 0
            # 位置情報のみを使用
            # （位置でオフセットをずらし、ワンホット値を設定）
            index = (d.pos_id * TABLE_NUM) + OneHotTable[d.card_id]
            v[index] = 1.0

    # 基本的なゲーム情報をセット
    game_data = input_data[1]
    v[BOARD_NUM + 0] = game_data.lp[0] / 8000.0  # 自分のLP
    v[BOARD_NUM + 1] = game_data.lp[1] / 8000.0  # 相手のLP
    assert game_data.current_phase >= constants.Phase.DRAW
    assert game_data.current_phase < constants.Phase.UPPER_VALUE
    v[BOARD_NUM + 2 + game_data.current_phase] = 1.0  # フェイズ

    # 現在の状況をセット
    request = input_data[4]
    index = BOARD_NUM + 2 + constants.Phase.UPPER_VALUE
    v[request.selection_type + 1 + index] = 1.0

    return v


def set_action_vector(input_data):
    """
    アクション情報をベクトル化する
    """
    action_data = input_data[5]
    n = len(action_data)

    v = np.zeros((n, ACTION_NUM), dtype=np.float32)
    offset = constants.CommandType.UPPER_VALUE + 1

    for i in range(n):
        # 各アクション情報をセット（サンプルでは一部の情報のみ使用します）
        act = action_data[i]
        # constants.CommandType.NO_VALUEも加味しておく
        v[i][act.command_type + 1] = 1.0

        # カード情報をセット
        card_id = act.card_id
        if card_id == constants.CardId.NO_VALUE:  # カードが無い場合（例：フェーズ移動）
            index = TABLE_NUM
        elif card_id == constants.CardId.UNKNOWN:  # カードが裏（伏せ）の場合
            index = TABLE_NUM + 1
        elif (
            card_id in OneHotTable
        ):  # カード情報がテーブルにある場合（アクションでは自分が持っていないカードも含まれる可能性があるため注意）
            index = OneHotTable[act.card_id]  # index < Table_Num
        else:  # テーブルに含まれないカード（カードプールが決定していれば、別のカードテーブルを作成しておくことは可能）
            index = TABLE_NUM + 2

        # カード情報をセット
        v[i][offset + index] = 1.0

    return v
