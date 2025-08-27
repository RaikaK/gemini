#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""デュエル中のカードによらない情報の定義"""

from .. import constants as c 
from dataclasses import dataclass

@dataclass
class GeneralData:
    """
    カードによらない情報
    """
    turn_num: int
    """0から始まるターン数"""
    current_phase: int
    """いまのフェイズ。ターンの切り替えタイミングではNullが来ることもある"""
    which_turn_now: int
    """どちらのターンか"""
    current_step: int
    """バトルフェイズのStepType"""
    current_damage_step: int
    """ダメージステップのDmgStepType"""
    lp: list[int]
    """お互いのライフポイント（[自分, 相手])）"""
    summon_num: list[int]
    """お互いの召喚可能回数（[自分, 相手]）"""

    def __init__(self, data):
        self.turn_num = data['turnNum']
        self.current_phase = c.Phase(data['currentPhase'])
        self.which_turn_now = c.PlayerId(data['whichTurnNow'])
        self.current_step = c.StepType(data['currentStep'])
        self.current_damage_step = c.DmgStepType(data['currentDamageStep'])
        self.lp = data['lp']
        self.summon_num = data['summonNum']

