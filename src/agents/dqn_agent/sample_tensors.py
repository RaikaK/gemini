import sys
import os

sys.path.append("C:/Users/b1/Desktop/u-ni-yo")

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

TableNum = len(OneHotTable)


# 盤面情報用ベクトル数（SetBoardVector関数で使用）
BoardNum = TableNum * constants.PosId.UPPER_VALUE
# ゲーム情報用ベクトル数（SetBoardVector関数で使用）
InforNum = 2 + constants.Phase.UPPER_VALUE + (constants.SelectionType.UPPER_VALUE + 1)
# アクション用ベクトル数（SetActionVector関数で使用）
ActionNum = (constants.CommandType.UPPER_VALUE + 1) + (TableNum + 3)


def SetBoardVector(Input):
    DuelTable = Input[2]
    n = len(DuelTable)

    v = np.zeros(BoardNum + InforNum, dtype=np.float32)

    # 自分のカードの盤面情報をセット（このサンプルでは相手の情報は見ません）
    for i in range(n):
        d: models.DuelCard = DuelTable[i]
        if d.player_id == constants.PlayerId.MYSELF:
            assert d.pos_id >= 0
            # 位置情報のみを使用
            # （位置でオフセットをずらし、ワンホット値を設定）
            index = (d.pos_id * TableNum) + OneHotTable[d.card_id]
            v[index] = 1.0

    # 基本的なゲーム情報をセット
    GameData = Input[1]
    v[BoardNum + 0] = GameData.lp[0] / 8000.0  # 自分のLP
    v[BoardNum + 1] = GameData.lp[1] / 8000.0  # 相手のLP
    assert GameData.current_phase >= constants.Phase.DRAW
    assert GameData.current_phase < constants.Phase.UPPER_VALUE
    v[BoardNum + 2 + GameData.current_phase] = 1.0  # フェイズ

    # 現在の状況をセット
    Request = Input[4]
    index = BoardNum + 2 + constants.Phase.UPPER_VALUE
    v[Request.selection_type + 1 + index] = 1.0

    return v


def SetActionVector(Input):
    ActionData = Input[5]
    n = len(ActionData)

    v = np.zeros((n, ActionNum), dtype=np.float32)
    Offset = constants.CommandType.UPPER_VALUE + 1

    for i in range(n):
        # 各アクション情報をセット（サンプルでは一部の情報のみ使用します）
        act = ActionData[i]
        # constants.CommandType.NO_VALUEも加味しておく
        v[i][act.command_type + 1] = 1.0

        # カード情報をセット
        CardId = act.card_id
        if CardId == constants.CardId.NO_VALUE:  # カードが無い場合（例：フェーズ移動）
            Index = TableNum
        elif CardId == constants.CardId.UNKNOWN:  # カードが裏（伏せ）の場合
            Index = TableNum + 1
        elif (
            CardId in OneHotTable
        ):  # カード情報がテーブルにある場合（アクションでは自分が持っていないカードも含まれる可能性があるため注意）
            Index = OneHotTable[act.card_id]  # Index < TableNum
        else:  # テーブルに含まれないカード（カードプールが決定していれば、別のカードテーブルを作成しておくことは可能）
            Index = TableNum + 2

        # カード情報をセット
        v[i][Offset + Index] = 1.0

    return v
