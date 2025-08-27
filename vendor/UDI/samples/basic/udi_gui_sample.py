#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""人間がguiを使ってUDIでデュエルを行う場合の実装.クライアント側は手動で起動する必要があります."""


from __future__ import annotations

import argparse
import queue
import sys
import time
import traceback

from ygo.udi_io import UdiIO
from ygo.gui.udi_gui_thread import UdiGUIThread


print(f"\n{sys.argv[0]} ver.1.0.0\n Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.\n")

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--connect", help="Set connect method. (default: Socket)", choices=["Socket", "gRPC"], default="Socket")
parser.add_argument("--tcp_host", help="Set TCP connect host name. (default: 127.0.0.1)", type=str, default="127.0.0.1")
parser.add_argument("--tcp_port", help="Set TCP connect port no. (default: 50001)", type=int, default=50001)

args = parser.parse_args()


def _mainloop(udi_io: UdiIO) -> None:
    global gui_thread
    global q

    try:
        if not udi_io.input():
            return

        # デュエル開始時の処理
        if udi_io.is_duel_start():
            print("デュエル開始")

        # デュエル終了時の処理
        if udi_io.is_duel_end():
            print("デュエル終了")
            print(udi_io.get_duel_end_data())
            # ログ出力
            print("ログを ./udi_logs/udi_gui に出力します")
            udi_io.flush_udi_logs()
            print("ログを出力しました")

        # 受け取ったデータをguiに反映
        gui_thread.set_data(duel_log_data=udi_io.get_duel_log_data(),
                            command_request=udi_io.get_command_request(),
                            duel_state_data=udi_io.get_duel_state_data())

        # 入力要求がある場合
        if udi_io.is_command_required():
            print("入力要求が来ています。GUIでコマンドを選択してください")
            # GUIアプリからのコマンド待ち
            while True:
                try:
                    command_index = q.get(block=False)
                except queue.Empty:
                    time.sleep(0.001)
                    continue

                command_index = int(command_index)
                commands = udi_io.get_commands()
                if 0 <= command_index < len(commands):
                    break
                time.sleep(0.001)

            udi_io.output_command(command_index)
            print(f"コマンド選択: {command_index}")

    except Exception:
        print('catch error:' + traceback.format_exc())

        print('exit python')

        exit()


# GUIの準備
q: queue.Queue = queue.Queue(1)     # コマンドをやり取りするためのキュー
gui_thread = UdiGUIThread()
gui_thread.start(q)

# クライアントを起動してもらう
print("DuelSimulatorかMDクライアントを起動してください")
print("シミュレータ起動コマンド例:")

print(
    "\n"
    "DuelSimulator.exe "
    "--tcp_port0 50001 "
    "--player_type0 Human "
    "--player_type1 CPU "
    "--deck_path0 DeckData/RoyaleBE.json "
    "--deck_path1 DeckData/RoyaleBE.json "
    "--loop_num 1 "
    "--randomize_seed true "
    "--log_level 3 "
    "--first_player 0 "
    "--lp0 8000 "
    "--lp1 8000 "
    "--hand_num0 5 "
    "--hand_num1 5 "
    "--seed 0 "
    "--play_reverse_duel false"
    "\n"
)

# UdiIOインスタンス準備
udi_io = UdiIO(tcp_host=args.tcp_host, tcp_port=args.tcp_port, connect=UdiIO.Connect.SOCKET if (args.connect == "Socket") else UdiIO.Connect.GRPC, api_version=1)
udi_io.start_udi_logging("./udi_logs/udi_gui") # ログ記録開始
udi_io.wait(_mainloop)

# EOF
