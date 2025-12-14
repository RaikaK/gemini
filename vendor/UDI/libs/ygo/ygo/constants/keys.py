#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""UdiIOが送られたjsonを直接参照するためのキー"""

class DuelData:
    COMMAND_REQUEST = "commandRequest"
    BOARD_DATA = "boardData"
    DUEL_LOG_DATA = "duelLogData"
    DUEL_STATE_DATA = "duelStateData"

class DuelStateData:
    DUEL_CARD_TABLE = "duelCardTable"
    GENERAL_DATA = "generalData"
    CHAIN_STACK = "chainStack"

class UdiLogData:
    COMMAND_REQUEST = "commandRequest"
    DUEL_LOG_DATA = "duelLogData"
    DUEL_STATE_DATA = "duelStateData"
    SELECTED_COMMAND = "selectedCommand"