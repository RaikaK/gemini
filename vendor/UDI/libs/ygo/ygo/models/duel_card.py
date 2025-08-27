#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""デュエル中のカード毎の情報の定義"""

from .. import constants as c
from dataclasses import dataclass

@dataclass
class DuelCard:
    """
    デュエル中のカード毎の情報。
    """
    card_id: int
    """カードのcardId"""
    player_id: int
    """場所のplayerId"""
    pos_id: int
    """場所のposId"""
    card_index: int
    """場所のcardIndex"""
    face: int
    """カードの表裏"""
    turn: int
    """カードの縦横（攻守）"""
    is_disabled: int
    """効果が無効になっているか（1なら無効になっている）"""
    atk_val: int
    """フィールドでの攻撃力"""
    def_val: int
    """フィールドでの守備力"""
    level: int
    """フィールドでのレベル"""
    is_attacking: int
    """攻撃中か（1なら攻撃中）"""
    is_attacked: int
    """攻撃対象に選択されているか（1なら選択されている）"""
    equip_target: int
    """このカードが装備されているカードのtableIndex"""
    magic_counter_num: int
    """乗っているカウンターの数"""
    used_effect1: int
    """このターン中①の効果を使ったかどうか（1なら使用済み）"""
    used_effect2: int
    """このターン中②の効果を使ったかどうか（1なら使用済み）"""
    used_effect3: int
    """このターン中③の効果を使ったかどうか（1なら使用済み）"""
    turn_passed: int
    """魔法＆罠ゾーン上で裏側で存在してから1ターン経過しているかどうか（1なら経過している）"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.card_index = data['cardIndex']
        self.face = c.Face(data['face'])
        self.turn = c.Turn(data['turn'])
        self.is_disabled = data['isDisabled']
        self.atk_val = data['atkVal']
        self.def_val = data['defVal']
        self.level = data['level']
        self.is_attacking = data['isAttacking']
        self.is_attacked = data['isAttacked']
        self.equip_target = data['equipTarget']
        self.magic_counter_num = data['magicCounterNum']
        self.used_effect1 = data['usedEffect1']
        self.used_effect2 = data['usedEffect2']
        self.used_effect3 = data['usedEffect3']
        self.turn_passed = data['turnPassed']