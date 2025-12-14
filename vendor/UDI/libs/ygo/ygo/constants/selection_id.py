#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""SelectionIdの定義"""

from enum import IntEnum

class SelectionId(IntEnum):
    """
    選択の説明をするid
    """
    NO_VALUE = -1
    """値無し"""
    SELECT_CARD_AS_TRIBUTE = 0
    """リリースするカードを選択してください。"""
    SET_A_SPELL_OR_TRAP_CARD_ON_THE_FIELD_Q = 1
    """手札から魔法・罠カードをセットしますか？"""
    SELECT_SPELL_OR_TRAP_CARD_TO_SET_ON_FIELD = 2
    """手札からセットする魔法・罠カードを選択してください。"""
    SELECT_MONSTER_TO_SPECIAL_SUMMON_FROM_YOUR_DECK = 3
    """デッキから特殊召喚するモンスターを選択してください。"""
    SELECT_CARD_TO_ADD_FROM_YOUR_DECK_TO_YOUR_HAND = 4
    """デッキから手札に加えるカードを選択してください。"""
    COIN_TOSS_SELECTION = 5
    """コインの裏表を選択してください"""
    SELECT_CARD_TO_SEND_TO_GRAVEYARD = 6
    """墓地へ送るカードを選択してください。"""
    SEND_A_CARD_IN_YOUR_HAND_TO_THE_GRAVEYARD = 7
    """手札のカードを墓地へ送ってください。"""
    CONTINUE_YOUR_MAIN_PHASE_Q = 8
    """メインフェイズを続けますか？"""
    SELECT_CARD_TO_RETURN_TO_DECK = 9
    """デッキに戻すカードを選択してください。"""
    SELECT_CARD_IN_GRAVEYARD_TO_BANISH = 10
    """墓地から除外するカードを選択してください。"""
    SELECT_CARD_TO_ADD_TO_YOUR_HAND = 11
    """手札に加えるカードを選択してください。"""
    SELECT_CARD_TO_TARGET = 12
    """対象とするカードを選択してください。"""
    SELECT_MONSTER_TO_EQUIP = 13
    """装備するモンスターを選択してください。"""
    SELECT_MONSTER = 14
    """モンスターを選択してください。"""
    SELECT_CARD_TO_DESTROY = 15
    """破壊するカードを選択してください。"""
    RETURN_THE_CARD_TO_THE_HAND_Q = 16
    """カードを手札に戻しますか？"""
    SELECT_MONSTER_YOUR_OPPONENT_CONTROLS_TO_TAKE_CONTROL = 17
    """コントロールを得る相手のモンスターを選択してください。"""
    SELECT_NECESSARY_MONSTER_TO_MATCH_REQUIRED_NUMBER_OF_LEVEL = 18
    """儀式召喚に必要なレベル分のモンスターを選択してください。"""
    TRIBUTE_NECESSARY_MONSTER_TO_MATCH_REQUIRED_NUMBER_OF_LEVEL = 19
    """儀式召喚に必要なレベル分のモンスターをリリースしてください。"""
    CONTINUE_TO_ATTACK_Q = 20
    """戦闘が巻き戻されました。攻撃を続けますか？"""
    USE_THE_EFFECT_Q = 21
    """効果を使用しますか？"""
    PAY_THE_COST_Q = 22
    """維持コストを払いますか？"""
    DISCARD_FROM_YOUR_HAND = 23
    """手札を捨ててください。"""
    SELECT_BATTLE_POSITION_OF_CARD = 24
    """表示形式を選択してください。"""
    USE_WHICH_EFFECT_Q = 25
    """どの効果を使用しますか？"""
    SELECT_CARD_TYPE = 26
    """カードの種類を選択してください。"""
    SELECT_MONSTER_TO_SWITCH_TO_FACEDOWN_DEFENSE_POSITION = 27
    """裏側守備表示にするモンスターを選択してください。"""
    REMOVE_COUNTER_FROM_YOUR_SIDE_OF_FIELD = 28
    """自分フィールド上のカウンターを取り除いてください。"""
    SELECT_MONSTER_TO_SPECIAL_SUMMON = 29
    """特殊召喚するモンスターを選択してください。"""
    SELECT_MONSTER_FROM_HAND_TO_SPECIAL_SUMMON = 30
    """特殊召喚するモンスターを手札から選択してください。"""
    SELECT_CARD_TO_PLACE_COUNTER_ON = 31
    """カウンターを置くカードを選択してください。"""
    SELECT_CARD_TO_RETURN_TO_HAND = 32
    """手札に戻すカードを選択してください。"""
    DESTROY_A_CARD_Q = 33
    """カードを破壊しますか？"""
    OTHER = 34
    """その他"""
    UPPER_VALUE = 35
    """上限値"""

    def __str__(self):
        descriptions = {
            SelectionId.NO_VALUE: "値無し",
            SelectionId.SELECT_CARD_AS_TRIBUTE: "リリースするカードを選択してください。",
            SelectionId.SET_A_SPELL_OR_TRAP_CARD_ON_THE_FIELD_Q: "手札から魔法・罠カードをセットしますか？",
            SelectionId.SELECT_SPELL_OR_TRAP_CARD_TO_SET_ON_FIELD: "手札からセットする魔法・罠カードを選択してください。",
            SelectionId.SELECT_MONSTER_TO_SPECIAL_SUMMON_FROM_YOUR_DECK: "デッキから特殊召喚するモンスターを選択してください。",
            SelectionId.SELECT_CARD_TO_ADD_FROM_YOUR_DECK_TO_YOUR_HAND: "デッキから手札に加えるカードを選択してください。",
            SelectionId.COIN_TOSS_SELECTION: "コインの裏表を選択してください",
            SelectionId.SELECT_CARD_TO_SEND_TO_GRAVEYARD: "墓地へ送るカードを選択してください。",
            SelectionId.SEND_A_CARD_IN_YOUR_HAND_TO_THE_GRAVEYARD: "手札のカードを墓地へ送ってください。",
            SelectionId.CONTINUE_YOUR_MAIN_PHASE_Q: "メインフェイズを続けますか？",
            SelectionId.SELECT_CARD_TO_RETURN_TO_DECK: "デッキに戻すカードを選択してください。",
            SelectionId.SELECT_CARD_IN_GRAVEYARD_TO_BANISH: "墓地から除外するカードを選択してください。",
            SelectionId.SELECT_CARD_TO_ADD_TO_YOUR_HAND: "手札に加えるカードを選択してください。",
            SelectionId.SELECT_CARD_TO_TARGET: "対象とするカードを選択してください。",
            SelectionId.SELECT_MONSTER_TO_EQUIP: "装備するモンスターを選択してください。",
            SelectionId.SELECT_MONSTER: "モンスターを選択してください。",
            SelectionId.SELECT_CARD_TO_DESTROY: "破壊するカードを選択してください。",
            SelectionId.RETURN_THE_CARD_TO_THE_HAND_Q: "カードを手札に戻しますか？",
            SelectionId.SELECT_MONSTER_YOUR_OPPONENT_CONTROLS_TO_TAKE_CONTROL: "コントロールを得る相手のモンスターを選択してください。",
            SelectionId.SELECT_NECESSARY_MONSTER_TO_MATCH_REQUIRED_NUMBER_OF_LEVEL: "儀式召喚に必要なレベル分のモンスターを選択してください。",
            SelectionId.TRIBUTE_NECESSARY_MONSTER_TO_MATCH_REQUIRED_NUMBER_OF_LEVEL: "儀式召喚に必要なレベル分のモンスターをリリースしてください。",
            SelectionId.CONTINUE_TO_ATTACK_Q: "戦闘が巻き戻されました。攻撃を続けますか？",
            SelectionId.USE_THE_EFFECT_Q: "効果を使用しますか？",
            SelectionId.PAY_THE_COST_Q: "維持コストを払いますか？",
            SelectionId.DISCARD_FROM_YOUR_HAND: "手札を捨ててください。",
            SelectionId.SELECT_BATTLE_POSITION_OF_CARD: "表示形式を選択してください。",
            SelectionId.USE_WHICH_EFFECT_Q: "どの効果を使用しますか？",
            SelectionId.SELECT_CARD_TYPE: "カードの種類を選択してください。",
            SelectionId.SELECT_MONSTER_TO_SWITCH_TO_FACEDOWN_DEFENSE_POSITION: "裏側守備表示にするモンスターを選択してください。",
            SelectionId.REMOVE_COUNTER_FROM_YOUR_SIDE_OF_FIELD: "自分フィールド上のカウンターを取り除いてください。",
            SelectionId.SELECT_MONSTER_TO_SPECIAL_SUMMON: "特殊召喚するモンスターを選択してください。",
            SelectionId.SELECT_MONSTER_FROM_HAND_TO_SPECIAL_SUMMON: "特殊召喚するモンスターを手札から選択してください。",
            SelectionId.SELECT_CARD_TO_PLACE_COUNTER_ON: "カウンターを置くカードを選択してください。",
            SelectionId.SELECT_CARD_TO_RETURN_TO_HAND: "手札に戻すカードを選択してください。",
            SelectionId.DESTROY_A_CARD_Q: "カードを破壊しますか？",
            SelectionId.OTHER: "その他",
            SelectionId.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")