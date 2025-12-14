#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""デュエルログの1つの単位の定義"""

from ..constants import DuelLogType 
from dataclasses import dataclass 


@dataclass
class DuelLogDataEntry:
    """
    デュエルログの一つの単位
    """
    type: DuelLogType
    """ログの種類"""
    data: dict
    """DuelLogTypeに応じた情報"""

    def __init__(self, data):
        self.type = DuelLogType(data['type'])
        self.data = data['data']