#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""ランダム返答を返す最小限な実装."""


from __future__ import annotations

import random
import sys
import traceback

from ygo.udi_io import UdiIO


print(f"\n{sys.argv[0]} ver.1.0.0\n Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.\n")


def _execute_random_command(udi_io: UdiIO, commands) -> bool:
    # ランダムにコマンド選択
    index = random.randrange(len(commands))

    # コマンド実行
    if (udi_io.output_command(index)):
        return True

    return False


def _mainloop(udi_io: UdiIO) -> None:
    try:
        if not udi_io.input():
            return

        # デュエル開始の情報かどうか
        if udi_io.is_duel_start():
            print("デュエル開始")

        # デュエル終了の情報かどうか
        if udi_io.is_duel_end():
            print("デュエル終了")
            print(udi_io.get_duel_end_data())

        # 入力要求に関する情報
        command_request = udi_io.get_command_request()

        # 実行可能なコマンドを取得（内部でcommand_requestを参照しています）
        commands = udi_io.get_commands()

        # デュエルログに関する情報
        duel_log_data = udi_io.get_duel_log_data()

        # 管理テーブルを含むデュエルの状況に関する情報
        duel_state_data = udi_io.get_duel_state_data()

        # 管理テーブル本体
        duel_card_table = udi_io.get_duel_card_table()

        # コマンド入力が要求されている場合ランダムなコマンドを実行
        if udi_io.is_command_required():
            _execute_random_command(udi_io, commands)
        else:
            return

    except Exception:
        print('catch error:', traceback.format_exc())

        print('exit python')

        sys.exit()


udi_io = UdiIO(api_version=1, tcp_host="10.95.102.79", tcp_port=50001)
udi_io.wait(_mainloop)

# EOF
