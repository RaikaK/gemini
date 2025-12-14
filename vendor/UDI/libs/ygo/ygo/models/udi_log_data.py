#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""UDIの情報を記録する形式の定義"""

from dataclasses import dataclass

from .command_request import CommandRequest 
from .duel_state_data import DuelStateData
from .duel_log_data_entry import DuelLogDataEntry

@dataclass
class UdiLogData:
    """
    udi_ioのログの保存フォーマット
    """
    command_request :CommandRequest
    """その時のcommand_request"""
    duel_state_data :DuelStateData
    """その時のduel_state_data"""
    duel_log_data :list[DuelLogDataEntry]
    """その時のduel_log_data"""
    selected_command :int
    """その時選択したコマンド"""