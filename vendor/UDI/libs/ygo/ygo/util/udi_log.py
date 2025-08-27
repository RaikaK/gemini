#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""UdiLogDataに関するUtil"""

import gzip
import json

from .. import models as mdl
from .. import constants as c

class UdiLogUtil:
    """UdiLogDataを扱うためのクラス"""

    @staticmethod
    def load_udi_log(file_path: str) -> list[mdl.UdiLogData]:
        """
        udi_logのファイルからログを読み込む。

        Parameters
        ----------
        file_path : str
            読み込みたい.gzファイルのパス

        Returns
        -------
        list[mdl.UdiLogData]
            読みこんだファイルのUdiLogDataのリスト
        """
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            json_str = f.read()
        data = json.loads(json_str)

        ret = []
        for entry in data:
            command_request = mdl.CommandRequest(entry[c.UdiLogData.COMMAND_REQUEST])
            duel_state_data = mdl.DuelStateData(entry[c.UdiLogData.DUEL_STATE_DATA])
            duel_log_data = [mdl.DuelLogDataEntry(d) for d in entry[c.UdiLogData.DUEL_LOG_DATA]]
            selected_command =  entry[c.UdiLogData.SELECTED_COMMAND]
            udi_log_data = mdl.UdiLogData(command_request, duel_state_data, duel_log_data, selected_command)
            ret.append(udi_log_data)

        return ret