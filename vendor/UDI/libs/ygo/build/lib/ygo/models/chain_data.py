#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""チェーンに関する情報の定義"""

from .. import constants as c 
from dataclasses import dataclass 

@dataclass
class ChainData:
    """
    チェーンに関する情報。（チェーンに積まれる前も含む）
    """
    table_index: int
    """このチェーンのカードのDuelCardTable上でのインデックス"""
    chain_num: int
    """このチェーンの番号（チェーン1は1）"""
    card_id: int
    """このチェーンのcardId"""
    effect_no: int
    """このチェーンの効果番号"""
    chain_state: int
    """このチェーンの状態(ChainState参照)"""
    target_table_index_list: list[int]
    """このチェーンで対象に取られているカードのDuelCardTable上のインデックスのリスト"""

    def __init__(self, data):
        self.table_index = data['tableIndex']
        self.chain_num = data['chainNum']
        self.card_id = data['cardId']
        self.effect_no = c.EffectNo(data['effectNo'])
        self.chain_state = c.ChainState(data['chainState'])
        self.target_table_index_list = data['targetTableIndexList']

