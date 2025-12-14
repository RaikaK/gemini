#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""
遊戯王UDI(Universal Duel Interface)との入出力.

UDIを実装したクライアント、具体的にはMaster DuelおよびDuel Simulatorとの通信しデータの入出力を行います。
"""

from __future__ import annotations

import copy
from datetime import datetime
import gzip
import json
import os
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent import futures
from enum import IntEnum, auto
from typing import Any

import grpc  # type: ignore

from . import udi_pb2
from . import udi_pb2_grpc
from . import constants as c
from . import models as mdl


class UdiIO:
    """UDI(Universal Duel Interface)用入出力クラス."""

    @staticmethod
    def __run_socket_server(udi_io: UdiIO, tcp_host: str, tcp_port: int) -> None:
        tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_server.bind((tcp_host, tcp_port))
        tcp_server.listen(10)
        sockname = tcp_server.getsockname()
        udi_io._tcp_port = sockname[1]
        udi_io._output_log(f"Start Socket Server: {sockname[0]}:{sockname[1]}")

        BUFSIZE = 1024

        raiseError = False  # noqa: F841
        udi_io._udi_length = -1

        while True:
            if (udi_io._socket_client is None):
                udi_io._output_log("connect to client ...")
                udi_io._socket_client, address = tcp_server.accept()
                udi_io._output_log("done.")

            data = b""

            while (udi_io._tcp_input != ""):
                time.sleep(0.0001)

            while True:
                received = False
                count = 0
                chunk = b""
                justsize = False
                while (not received):
                    try:
                        chunk = udi_io._socket_client.recv(BUFSIZE + 1, socket.MSG_PEEK)
                        justsize = (len(chunk) == BUFSIZE)
                        chunk = udi_io._socket_client.recv(BUFSIZE)
                        received = True
                    except ConnectionResetError as e:
                        if (count < 10):
                            count += 1
                            udi_io._output_log(f"retry: {e}")
                        else:
                            # 10回やってもダメなら諦め
                            raise e
                    except ConnectionAbortedError as e:
                        udi_io._output_log(f"Already aborted: {e}")
                        return

                data += chunk
                if (0 < udi_io._udi_length):
                    if (len(data) < udi_io._udi_length):
                        pass

                    else:
                        udi_io._udi_length = -1
                        break

                elif (justsize or (len(chunk) < BUFSIZE)):
                    break

            """
            # 1/2でエラーを起こしてみる
            if raiseError:
                data = b"{" + data
                raiseError = False
            else:
                raiseError = True
            """

            try:
                dj = json.loads(data.decode("utf-8"))
                response = udi_io._data_json_to_udi(dj)
                udi_io._socket_client.sendall(response.encode("utf-8"))
                if (response == "udiexit"):
                    udi_io._output_log("udiexit. Now socket is None")
                    udi_io._socket_client = None

            except json.JSONDecodeError:
                udi_io._output_log("JSONDecodeError")
                udi_io._output_log(traceback.format_exc())
                udi_io._output_log(str(len(data)) + " bytes :" + str(data))

            except Exception:
                udi_io._output_log(traceback.format_exc())
                udi_io._output_log(str(data))

    class __UdiRpc(udi_pb2_grpc.UdiRpcServicer):

        def __init__(self, udi_io: UdiIO) -> None:
            self.udi_io = udi_io

        def SendDuel(self, request: udi_pb2.DuelData, context) -> udi_pb2.DuelReply:
            reply = "Unknown"

            while (self.udi_io._tcp_input != ""):
                time.sleep(0.0001)

            try:
                dj = json.loads(request.data)
                reply = self.udi_io._data_json_to_udi(dj)

            except json.JSONDecodeError:
                reply = "JSONDecodeError"
                self.udi_io._output_log("JSONDecodeError")
                self.udi_io._output_log(traceback.format_exc())
                self.udi_io._output_log(str(len(request.data)) + " bytes :" + request.data)

            except Exception:
                self.udi_io._output_log(traceback.format_exc())
                self.udi_io._output_log(request.data)

            return udi_pb2.DuelReply(data=reply)

    class Connect(IntEnum):
        """UDIで使う接続方法."""

        SOCKET = auto()
        """
        Socket接続.

        UnityやMDクライアントに繋ぐ場合は必ずこちらを指定してください。
        """
        GRPC = auto()
        """
        gRPC接続.

        クライアントがDuel Simulatorの時のみ使用可能。長時間/大量接続等の時の耐久度はこちらが高いようです
        """

    class OutputOptionalDataType(IntEnum):
        """
        オプションのデータタイプ.

        デュエルするだけなら必要無いけど追加表示を出すのに使ったりするデータの種類
        """

        ETC = 0
        """
        その他.

        まだどう使うか決まってないデータなどをとりあえず送る時に使用
        """
        GUESS_SET_CARD = 1
        """セットカード予測."""
        RATING_TEXT = 2
        """評価値テキスト."""
        SITUATION_SCORE = 3
        """形勢表示用スコア."""

    class RatingTextType(IntEnum):
        """
        RatingTextのデータタイプ.

        選択肢に対応するサムネイルの指定をするタイプ
        c#側のRatingTextJson.Typeと一致している必要がある
        """

        ETC = 0
        """
        その他.

        まだどう使うか決まってないデータなどをとりあえず送る時に使用
        """
        CARD_ID = 1
        """カードID."""
        PHASE = 2
        """フェイズ."""
        POSITION = 3
        """盤面のposId"""
        CANCEL = 4
        """キャンセル.（idに意味なし）"""
        DRAW = 5
        """ドロー.（idに意味なし）"""
        COIN = 6
        """コインの表裏."""

    def __init__(self, *, tcp_host: str = "127.0.0.1", tcp_port: int = 8573, tcpport: int = -1, connect: Connect = Connect.SOCKET, api_version: int = 0) -> None:
        """
        UDI(Universal Duel Interface)用入出力クラス.

        Parameters
        ----------
        tcp_host :
            接続に使うTCPホスト名
        tcp_port :
            接続に使うTCPポート番号。0を指定した場合は空いてるポートを自動割り当て
        tcpport :
            tcp_portの旧名。廃止予定
        connect :
            接続方式
        api_version :
            使用するAPIバージョン。現状0か1

        Attributes
        ----------
        duel_data : dict[str, Any]
            UDIから取得したデュエルに関する情報が入った辞書
        """
        self.__api_version = api_version

        self.__requests: list[dict[str, Any]] = []
        self._tcp_input = ""

        self.__duel_data: dict[str, Any] = {}

        self.__duel_log_data: list[dict[str, Any]] = []

        self.output_log: Callable[[str], None] | None = None
        self.log_response_history = True

        self._udi_logging = False
        self._udi_log_buf: list = []
        self._udi_log_duel_count = 0

        self._tcp_host = tcp_host
        self._tcp_port = -1
        self.__thread_lock = threading.Lock()
        self._udi_length = -1
        self._socket_client: socket.socket | None = None
        self.__grpc_server = None
        self.mainloop = None

        if (0 <= tcpport):
            tcp_port = tcpport

        if (connect == UdiIO.Connect.SOCKET):
            thread = threading.Thread(target=UdiIO.__run_socket_server, args=(self, tcp_host, tcp_port), daemon=True)
            thread.start()

            while self.tcp_port < 0:
                time.sleep(0.0001)

            self.output(sys.version)
            self.output('defaultencoding:' + sys.getdefaultencoding())
            self.output(os.getcwd())

        elif (connect == UdiIO.Connect.GRPC):
            self.__grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
            udi_pb2_grpc.add_UdiRpcServicer_to_server(UdiIO.__UdiRpc(self), self.__grpc_server)
            self._tcp_port = self.__grpc_server.add_insecure_port(f"{tcp_host}:{tcp_port}")
            self.__grpc_server.start()
            print(f"Start gRPC Server: {tcp_host}:{self.tcp_port}")

    @property
    def duel_data(self) -> dict[str, Any]:
        """UDIから取得したデュエルに関する情報が入った辞書."""
        return self.__duel_data

    @property
    def tcp_host(self) -> str:
        """接続済みTCPホスト名."""
        return self._tcp_host

    @property
    def tcp_port(self) -> int:
        """接続済みTCPポート番号."""
        return self._tcp_port

    @property
    def api_version(self) -> int:
        """APIバージョン."""
        return self.__api_version

    def wait(self, mainloop: Callable[[UdiIO], None]) -> None:
        """
        待ちループに入る.

        一度入ったら基本帰ってきません

        Parameters
        ----------
        mainloop :
            UDIとデータ入出力するメインループを指定
        """
        while True:
            mainloop(self)
            time.sleep(0.001)

    def quit(self) -> None:
        """
        終了処理.

        とりあえずgRPCのスレッドを明示的に片付けておく
        """
        self._tcp_input = ""
        if (self._socket_client is not None):
            self._socket_client.close()

        if (self.__grpc_server is not None):
            self.__grpc_server.stop(None)

    def input(self) -> bool:
        """
        UDIからのデータ入力.

        データが返ってくるまでブロックします

        Returns
        -------
        入力データがあったら True。無かったらFalse。
        Falseが帰ってきたら終了の合図です
        """
        input_val = ""

        wait_input = True
        while wait_input:
            with self.__thread_lock:
                if (0 < len(self._tcp_input)):
                    input_val += self._tcp_input
                    self._tcp_input = ""
                    wait_input = False

            if (wait_input):
                time.sleep(0.0001)

        if input_val:
            # 終了の指示が来たら強制で終了する
            if (input_val == 'udiexit'):
                self.flush_udi_logs()
                self.__duel_log_data = []
                return False

            if (input_val != 'readonly'):
                self.__duel_data = json.loads(input_val)

                self._add_duel_log_data()

                self._add_udi_log()

        return True

    def output(self, data: dict[str, Any] | str) -> None:
        """
        UDIへのレスポンス出力.

        コマンド等のレスポンスを返します

        Parameters
        ----------
        data :
            レスポンスデータ
        """
        with self.__thread_lock:
            if (isinstance(data, dict)):
                self.__requests.append(data)
            else:
                self.__requests.append({"header": data})

    def _data_json_to_udi(self, dj: dict[str, Any]) -> str:
        response = "Unknown"
        skip_log = False

        with self.__thread_lock:
            if ("commandRequest" in dj):
                self._tcp_input = json.dumps(dj)
                response = "Ok"

            elif (("udi_length" in dj)):
                self._udi_length = dj["udi_length"]
                response = f"udi_length: {self._udi_length}"

            elif (not ("udi_type" in dj)):
                pass

            elif (dj["udi_type"] == "request"):
                if (len(self.__requests) == 0):
                    response = json.dumps({"header": "empty requests"})
                    skip_log = True

                else:
                    request = self.__requests.pop(0)
                    if (request["header"] != "udi"):
                        skip_log = True
                    response = json.dumps(request)

            elif (dj["udi_type"] == "udiexit"):
                self._tcp_input = "udiexit"
                response = "udiexit"

            else:
                self._tcp_input = dj["udi_type"]
                response = f"received: {dj['udi_type']}"

        if (self.log_response_history and (not skip_log)):
            self._output_log(f"response: {response}")

        return response

    def _output_log(self, msg: str) -> None:
        if (self.output_log is None):
            self.output(msg)
            print(msg)
        else:
            self.output_log(msg)

    def _add_duel_log_data(self) -> None:
        if (self.is_duel_start()):
            self.__duel_log_data = []

        self.__duel_log_data += self.__duel_data[c.DuelData.DUEL_LOG_DATA]

    def output_command(self, index: int) -> bool:
        """
        選択したコマンドを出力。CommandRequestのcommands内のインデックスで指定する

        Parameters
        ----------
        index :
            選択したコマンドのインデックス

        Returns
        -------
        出力成功ならTrue
        パラメータの範囲や数等が合っていれば基本大丈夫なはずです
        """
        self._set_selected_command(index)
        self.output({"header": "udi", "index": index})
        return True

    def output_optional_data(self, data_type: UdiIO.OutputOptionalDataType, optional_data: dict[str, Any]) -> bool:
        """
        オプションのデータを出力.

        デュエルするだけなら必要無いけど追加表示を出すのに使ったりするデータを出力する
        """
        self.output({**{"header": "udi_optional_data", "optional_data_type": data_type}, **optional_data})
        return True

    def output_guess_set_card(self, entries: list[dict[str, int | float]]) -> bool:
        """
        セットカード予測データの出力.

        Parameters
        ----------
        entries :
            予測データのリスト。各エントリの形式は {"player": プレイヤー番号, "position": 場所, "cardId": カードID, "score": 確率}
        """
        return self.output_optional_data(UdiIO.OutputOptionalDataType.GUESS_SET_CARD, {"entries": entries})

    def output_rating_text(self, entries: list[dict[str, str | float]]) -> bool:
        """
        評価値テキストデータの出力.

        Parameters
        ----------
        entries :
            評価値エントリのリスト。各エントリの形式は {"text": テキスト, "score": 評価値(0.0-1.0), "type": サムネイル用のタイプ(RatingTextType), "id":typeに応じたサムネイル用id(int)}
        """
        # 旧仕様互換
        for entry in entries:
            if "cardId" in entry:
                entry["type"] = UdiIO.RatingTextType.CARD_ID
                entry["id"] = entry["cardId"]
                entry.pop("cardId")

        return self.output_optional_data(UdiIO.OutputOptionalDataType.RATING_TEXT, {"entries": entries})

    def output_situation_score(self, entries: list[dict[str, str | float]]) -> bool:
        """
        形勢評価スコアの出力.

        Parameters
        ----------
        entries :
            形勢評価スコアエントリのリスト。各エントリの形式は {"score": 評価値(0.0-1.0), "player": プレイヤー(第三者視点なら-1、自分視点なら0)}
        """
        return self.output_optional_data(UdiIO.OutputOptionalDataType.SITUATION_SCORE, {"entries": entries})

    def is_duel_start(self) -> bool:
        """
        現在のデータがデュエル開始を表すかを返す
        デュエル開始のタイミングでのみtrueになる

        Returns
        -------
        bool
            デュエル開始ならtrue。それ以外ならfalse
        """
        new_duel_log_data = self.get_new_duel_log_data()
        if not new_duel_log_data:
            return False

        current_log = new_duel_log_data[-1]
        is_duel_start = current_log.type == c.DuelLogType.DUEL_START
        return is_duel_start

    def is_duel_end(self) -> bool:
        """
        現在のデータがデュエル終了を表すかを返す
        デュエル終了のタイミングでのみtrueになる

        Returns
        -------
        bool
            デュエル終了ならtrue
        """
        new_duel_log_data = self.get_new_duel_log_data()
        if not new_duel_log_data:
            return False

        current_log = new_duel_log_data[-1]
        is_duel_end = current_log.type == c.DuelLogType.DUEL_END
        return is_duel_end

    def get_duel_end_data(self) -> mdl.DuelEndData | None:
        """
        デュエル終了時の情報を返す

        Returns
        -------
        mdl.DuelEndData | None
            デュエル終了時の情報(model.DuelEndData)
            デュエル終了時以外はNone
        """
        duel_log_data = self.get_new_duel_log_data()
        current_log = duel_log_data[-1]
        duel_end_data = None
        if current_log.type == c.DuelLogType.DUEL_END:
            duel_end_data = mdl.DuelEndData(current_log.data)
        return duel_end_data

    def is_command_required(self) -> bool:
        """
        入力要求が来ているかどうかを返す

        Returns
        -------
        bool
            コマンド入力が要求されているならtrue
        """
        commands = self.get_commands()
        is_command_required = len(commands) > 0
        return is_command_required

    def get_command_request(self) -> mdl.CommandRequest:
        """
        入力要求に関する情報CommandRequestを取得する

        Returns
        -------
        mdl.CommandRequest
            入力要求に関する情報
        """
        data = self.__duel_data[c.DuelData.COMMAND_REQUEST]
        command_request = mdl.CommandRequest(data)
        return command_request

    def get_commands(self) -> list[mdl.CommandEntry]:
        """
        入力データから入力要求のコマンドのリストを取得する

        Returns
        -------
        list[mdl.CommandEntry]
            入力要求に含まれるコマンドのリスト
        """
        command_request = self.get_command_request()
        return command_request.commands

    def get_command_log(self) -> list[mdl.CommandLogEntry]:
        """
        CommandRequestのCommandLogを取得する

        Returns
        -------
        list[mdl.CommandLogEntry]
            CommandLog
        """
        command_request = self.get_command_request()
        return command_request.command_log

    def get_duel_log_data(self) -> list[mdl.DuelLogDataEntry]:
        """
        入力データからデュエルのログの情報DuelLogDataを取得する
        udi_ioで貯めているデュエル開始からのログ全てを取得する

        Returns
        -------
        list[mdl.DuelLogDataEntry]
            デュエル開始からのデュエルログ情報
        """
        return [mdl.DuelLogDataEntry(d) for d in self.__duel_log_data]

    def get_new_duel_log_data(self) -> list[mdl.DuelLogDataEntry]:
        """
        入力データで新たに追加されたデュエルのログの情報を取得

        Returns
        -------
        list[mdl.DuelLogDataEntry]
            新たに追加されたデュエルログ情報
        """
        return [mdl.DuelLogDataEntry(d) for d in self.__duel_data[c.DuelData.DUEL_LOG_DATA]]

    def get_duel_state_data(self) -> mdl.DuelStateData:
        """
        デュエルの現在の情報DuelStateDataを取得

        Returns
        -------
        mdl.DuelStateData
            デュエルの現在の情報
        """
        duel_state_data = mdl.DuelStateData(self.__duel_data[c.DuelData.DUEL_STATE_DATA])
        return duel_state_data

    def get_duel_card_table(self) -> list[mdl.DuelCard]:
        """
        DuelStateDataのカード毎の情報DuelCardTableを取得

        Returns
        -------
        list[dict[str, Any]]
            デュエル中のカード毎の情報
        """
        duel_state_data = self.get_duel_state_data()
        return duel_state_data.duel_card_table

    def get_chain_stack(self) -> list[mdl.ChainData]:
        """
        DuelStateDataのデュエル中のチェーンの情報ChainStackを取得

        Returns
        -------
        list[dict[str, Any]]
            デュエル中のチェーンの情報
        """
        duel_state_data = self.get_duel_state_data()
        return duel_state_data.chain_stack

    def start_udi_logging(self, log_dir="./udi_log") -> None:
        """
        UdiLogの記録を開始する

        Parameters
        ----------
        log_dir : str, optional
            ログを出力するディレクトリ, by default "./udi_log"
        """
        self._udi_logging = True
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self._log_dir = log_dir
        self._udi_log_buf = []

    def flush_udi_logs(self) -> None:
        """
        バッファのログをファイルに出力する
        """
        if not self._udi_logging or not self._udi_log_buf:
            return

        date_str = datetime.now().strftime("%y%m%d_%H%M%S")
        file_name = f"udi_log_{date_str}.json.gz"

        json_str = json.dumps(self._udi_log_buf)
        binary_data = json_str.encode("utf-8")
        with gzip.open(f"{os.path.join(self._log_dir, file_name)}", mode="wb") as f:
            f.write(binary_data)

        self._udi_log_buf = []
        self._udi_log_duel_count = 0

    def _add_udi_log(self) -> None:
        """
        バッファにログを追加する（10デュエルたまったら強制的に出力する）
        """
        if self._udi_logging:
            log_entry = {
                f"{c.UdiLogData.COMMAND_REQUEST}": self.__duel_data[c.DuelData.COMMAND_REQUEST],
                f"{c.UdiLogData.DUEL_STATE_DATA}": self.__duel_data[c.DuelData.DUEL_STATE_DATA],
                f"{c.UdiLogData.DUEL_LOG_DATA}": copy.deepcopy(self.__duel_log_data),
                f"{c.UdiLogData.SELECTED_COMMAND}": -1,
            }
            self._udi_log_buf.append(log_entry)

            # 10デュエルたまった場合は強制的にフラッシュ
            if self.is_duel_end():
                self._udi_log_duel_count += 1
                if self._udi_log_duel_count >= 10:
                    self.flush_udi_logs()

    def _set_selected_command(self, selected_command: int) -> None:
        """
        udiがコマンドを選択した時に最新のログのselectedCommandを更新する

        Parameters
        ----------
        selected_command : int
            選択したコマンド
        """
        if self._udi_logging:
            if len(self._udi_log_buf) > 0:
                self._udi_log_buf[-1][c.UdiLogData.SELECTED_COMMAND] = selected_command

# EOF
