#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""ランダム返答を返す実装同士で戦わせるのを1スクリプトでやってみたバージョン."""


import argparse
import datetime
import os
import pprint
import random
import sys
import time
import threading
import traceback

import ygo.constants as const
import ygo.models as mdl
import ygo.util.text as util_text
from ygo.udi_io import UdiIO


print(f"\n{sys.argv[0]} ver.1.0.0\n Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.\n")

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--connect", help="Set connect method. (default: Socket)", choices=["Socket", "gRPC"], default="Socket")
parser.add_argument("--tcp_host", help="Set TCP connect host name. (default: 127.0.0.1)", type=str, default="127.0.0.1")
parser.add_argument("--tcp_port0", help="Set TCP connect port no for player 0. (default: 50001)", type=int, default=50001)
parser.add_argument("--tcp_port1", help="Set TCP connect port no for player 1. (default: 50002)", type=int, default=50002)
parser.add_argument("--randomseed", help="Fix random seed to ...", type=int)

args = parser.parse_args()

if (args.randomseed is not None):
    random.seed(args.randomseed)

logfs = None
input_fs = None


def _output_log(log: str) -> None:
    global udi_io
    global logfs

    print(log, flush=True)

    if (logfs is not None):
        logfs.write(log + "\n")
        logfs.flush()


def _output_player_log(player_no: int, log: str) -> None:
    _output_log(f"[{player_no}]: {log}")


def _execute_random_command(player_no: int, udi_io: UdiIO, text_util: util_text.TextUtil) -> bool:
    # 実行可能なコマンドを取得
    commands = udi_io.get_commands()
    for i, command in enumerate(commands):
        _output_player_log(player_no, f"command({i}): {text_util.get_command_entry_text(command)}")

    # ランダムにコマンド選択
    index = random.randrange(len(commands))
    _output_player_log(player_no, f"\nselected {index}: {text_util.get_command_entry_text(commands[index])}\n")

    score = random.random()
    if score < 0.5:
        # 評価値テキスト送信のテスト。別にここで呼ぶ必要はなくてコマンド送信より手前ならどこでもいいです
        udi_io.output_rating_text([
            {"text": "[3]センジュ・ゴッド: 召喚", "score": 0.98, "cardId": 1020},
            {"text": "バトルフェイズへ移行", "score": 0.50, "type": UdiIO.RatingTextType.PHASE, "id": 3},
            {"text": "メインフェイズ2へ移行", "score": 0.50, "type": UdiIO.RatingTextType.PHASE, "id": 4},
            {"text": "裏", "score": 0.50, "type": UdiIO.RatingTextType.COIN, "id": 0},
            {"text": "中央のモンスターゾーン", "score": 0.35, "type": UdiIO.RatingTextType.POSITION, "id": 2}
        ])
    else:
        udi_io.output_rating_text([
            {"text": "キャンセル", "score": 0.98, "type": UdiIO.RatingTextType.CANCEL, "id": 0},
            {"text": "青眼の白龍", "score": 0.50, "type": UdiIO.RatingTextType.CARD_ID, "id": 1006},
            {"text": "ドロー", "score": 0.50, "type": UdiIO.RatingTextType.DRAW, "id": 0},
            {"text": "表", "score": 0.50, "type": UdiIO.RatingTextType.COIN, "id": 1},
            {"text": "左端の魔法罠ゾーン", "score": 0.35, "type": UdiIO.RatingTextType.POSITION, "id": 7}
        ])

    # コマンド実行
    if (udi_io.output_command(index)):
        return True

    return False


def _getnow() -> datetime.datetime:
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    return datetime.datetime.now(JST)


def _pick_deck_cards(cards: list[mdl.DuelCard], text_util: util_text.TextUtil) -> dict:
    ret: dict = {}
    ret[const.PlayerId.MYSELF] = {const.PosId.DECK: [], const.PosId.EXTRA: []}
    ret[const.PlayerId.RIVAL] = {const.PosId.DECK: [], const.PosId.EXTRA: []}
    for card in cards:
        if (card.card_id == const.CardId.NO_VALUE):
            continue
        if ((card.pos_id != const.PosId.DECK) and (card.pos_id != const.PosId.EXTRA)):
            continue

        ret[card.player_id][card.pos_id].append(text_util.get_card_name(card.card_id))

    return ret


def _update_board_score(udi_io: UdiIO) -> None:
    udi_io.output_situation_score([{"score": random.random()}])


def _guess_set_card(player_no: int, udi_io: UdiIO) -> None:
    req = udi_io.get_command_request()
    if (req.selection_type == const.SelectionType.DRAW_PHASE):
        pass
    elif (req.selection_type == const.SelectionType.MAIN_PHASE):
        pass
    elif (req.selection_type == const.SelectionType.BATTLE_PHASE):
        pass
    else:
        return

    dummy_card_ids = (1013, 1014, 1015, 1030, 1031)
    entries = []
    for card in udi_io.get_duel_card_table():
        if (card.player_id == const.PlayerId.MYSELF):
            continue
        if (card.face != const.Face.BACK):
            continue
        if ((card.pos_id != const.PosId.FIELD) and (card.pos_id != const.PosId.MAGIC_L_L) and (card.pos_id != const.PosId.MAGIC_L) and (card.pos_id != const.PosId.MAGIC_C) and (card.pos_id != const.PosId.MAGIC_R) and (card.pos_id != const.PosId.MAGIC_R_R)):
            continue

        cids = random.sample(dummy_card_ids, 3)
        score_left = 1.0
        for cid in cids:
            score = (random.random() * 0.6 + 0.3) * score_left
            score_left -= score
            entries.append({"player": card.player_id, "position": card.pos_id, "cardId": cid, "score": score})

    if (0 < len(entries)):
        _output_player_log(player_no, "guess_set_card: " + pprint.pformat(entries, width=160, compact=True))
        udi_io.output_guess_set_card(entries)


def _wait_loop(player_no: int, udi_io: UdiIO, text_util: util_text.TextUtil) -> None:
    while True:
        try:
            if not udi_io.input():
                _output_player_log(player_no, "input returns False")
                continue

            # 管理テーブル本体
            duel_card_table = udi_io.get_duel_card_table()

            # デュエル開始の情報かどうか
            if udi_io.is_duel_start():
                _output_player_log(player_no, "DuelStart - card: " + pprint.pformat(_pick_deck_cards(duel_card_table, text_util), width=160, compact=True))

            # デュエル終了の情報かどうか
            if udi_io.is_duel_end():
                _output_player_log(player_no, "デュエル終了")
                _output_player_log(player_no, str(udi_io.get_duel_end_data()))

            # 入力要求に関する情報
            # command_request = udi_io.get_command_request()

            # デュエルログに関する情報
            # duel_log_data = udi_io.get_duel_log_data()

            # 管理テーブルを含むデュエルの状況に関する情報
            # duel_state_data = udi_io.get_duel_state_data()

            # 盤面スコア更新
            _update_board_score(udi_io)

            # セットカード予測送信のテスト。別にここで呼ぶ必要はなくてコマンド送信より手前ならどこでもいいです
            _guess_set_card(player_no, udi_io)

            # コマンド入力が要求されている場合ランダムなコマンドを実行
            if udi_io.is_command_required():
                _execute_random_command(player_no, udi_io, text_util)
            else:
                continue

        except Exception:
            print('catch error:' + traceback.format_exc())

            print('exit python')

            exit()

        time.sleep(0.001)


text_util = util_text.TextUtil()

udi_io0 = UdiIO(tcp_host=args.tcp_host, tcp_port=args.tcp_port0, connect=UdiIO.Connect.SOCKET if (args.connect == "Socket") else UdiIO.Connect.GRPC, api_version=1)
udi_io0.output_log = _output_log

udi_io1 = UdiIO(tcp_host=args.tcp_host, tcp_port=args.tcp_port1, connect=UdiIO.Connect.SOCKET if (args.connect == "Socket") else UdiIO.Connect.GRPC, api_version=1)
udi_io1.output_log = _output_log

os.makedirs('./UDI/log', exist_ok=True)

d = _getnow().strftime('%Y%m%d%H%M%S')
with open("./UDI/log/udi_" + d + ".log", mode='w', encoding='UTF-8') as fs:
    logfs = fs
    with open("./UDI/log/udi_input_" + d + ".log", mode='w', encoding='UTF-8') as fs2:
        input_fs = fs2
        threading.Thread(target=_wait_loop, args=(0, udi_io0, text_util), daemon=True).start()
        threading.Thread(target=_wait_loop, args=(1, udi_io1, text_util), daemon=True).start()

        while True:
            time.sleep(0.001)

# EOF
