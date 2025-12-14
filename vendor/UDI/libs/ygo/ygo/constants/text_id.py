#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""TextIdの定義"""

from enum import IntEnum

class TextId(IntEnum):
    """
    デュエル中のテキストを取得するためのid
    """
    NO_VALUE = -1
    """値無し"""
    WAS_SELECTED = 0
    """「%s」が選択されました。"""
    DUE_TO_THE_CARDS_EFFECT_TURN_IS_SKIPPED = 1
    """カード効果によりターンがスキップされます。"""
    DUE_TO_THE_CARDS_EFFECT_DRAW_PHASE_IS_SKIPPED = 2
    """カード効果によりドローフェイズがスキップされます。"""
    DUE_TO_THE_CARDS_EFFECT_STANDBY_PHASE_IS_SKIPPED = 3
    """カード効果によりスタンバイフェイズがスキップされます。"""
    DUE_TO_THE_CARDS_EFFECT_MAIN_PHASE1_IS_SKIPPED = 4
    """カード効果によりメインフェイズ１がスキップされます。"""
    DUE_TO_THE_CARDS_EFFECT_BATTLE_PHASE_IS_SKIPPED = 5
    """カード効果によりバトルフェイズがスキップされます。"""
    DUE_TO_THE_CARDS_EFFECT_MAIN_PHASE2_IS_SKIPPED = 6
    """カード効果によりメインフェイズ２がスキップされます。"""
    FUSION_MONSTER = 7
    """融合モンスター"""
    SYNCHRO_MONSTER = 8
    """シンクロモンスター"""
    XYZ_MONSTER = 9
    """エクシーズモンスター"""
    RITUAL_MONSTER = 10
    """儀式モンスター"""
    PENDULUM_MONSTER = 11
    """ペンデュラムモンスター"""
    HEADS = 12
    """表"""
    TAILS = 13
    """裏"""
    BOTH_EFFECTS = 14
    """両方の効果"""
    SPELL_CARDS = 15
    """魔法カード"""
    TRAP_CARDS = 16
    """罠カード"""
    OK = 17
    """OK"""
    OTHER = 18
    """その他"""
    UPPER_VALUE = 19
    """上限値"""
    def __str__(self):
        descriptions = {
            TextId.NO_VALUE: "値無し",
            TextId.WAS_SELECTED: "「%s」が選択されました。",
            TextId.DUE_TO_THE_CARDS_EFFECT_TURN_IS_SKIPPED: "カード効果によりターンがスキップされます。",
            TextId.DUE_TO_THE_CARDS_EFFECT_DRAW_PHASE_IS_SKIPPED: "カード効果によりドローフェイズがスキップされます。",
            TextId.DUE_TO_THE_CARDS_EFFECT_STANDBY_PHASE_IS_SKIPPED: "カード効果によりスタンバイフェイズがスキップされます。",
            TextId.DUE_TO_THE_CARDS_EFFECT_MAIN_PHASE1_IS_SKIPPED: "カード効果によりメインフェイズ１がスキップされます。",
            TextId.DUE_TO_THE_CARDS_EFFECT_BATTLE_PHASE_IS_SKIPPED: "カード効果によりバトルフェイズがスキップされます。",
            TextId.DUE_TO_THE_CARDS_EFFECT_MAIN_PHASE2_IS_SKIPPED: "カード効果によりメインフェイズ２がスキップされます。",
            TextId.FUSION_MONSTER: "融合モンスター",
            TextId.SYNCHRO_MONSTER: "シンクロモンスター",
            TextId.XYZ_MONSTER: "エクシーズモンスター",
            TextId.RITUAL_MONSTER: "儀式モンスター",
            TextId.PENDULUM_MONSTER: "ペンデュラムモンスター",
            TextId.HEADS: "表",
            TextId.TAILS: "裏",
            TextId.BOTH_EFFECTS: "両方の効果",
            TextId.SPELL_CARDS: "魔法カード",
            TextId.TRAP_CARDS: "罠カード",
            TextId.OK: "OK",
            TextId.OTHER: "その他",
            TextId.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")