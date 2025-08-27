#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""デュエルの現在の情報の定義"""

from dataclasses import dataclass 
from .duel_card import DuelCard
from .general_data import GeneralData
from .chain_data import ChainData

@dataclass
class DuelStateData:
    """
    デュエルの現在の情報
    """
    duel_card_table: list[DuelCard]
    """カード毎の情報"""
    general_data: GeneralData
    """カードによらないデュエルの情報"""
    chain_stack: list[ChainData]
    """チェーンに関する情報"""

    def __init__(self, data):
        self.duel_card_table = [DuelCard(d) for d in data['duelCardTable']]
        self.general_data = GeneralData(data['generalData'])
        self.chain_stack = [ChainData(d) for d in data['chainStack']]

