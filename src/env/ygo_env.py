from queue import Queue
import time
from typing import cast

from ygo.constants import FinishType, ResultType
from ygo.models import DuelEndData
from ygo.models.command_request import CommandRequest
from ygo.models.duel_log_data_entry import DuelLogDataEntry
from ygo.models.duel_state_data import DuelStateData
from ygo.udi_io import UdiIO

from src.env.action_data import ActionData
from src.env.state_data import StateData
from src.gui.gui_thread import GUIThread


class YgoEnv:
    """
    遊戯王のGym環境
    """

    def __init__(self, tcp_host: str, tcp_port: int, use_grpc: bool, use_gui: bool) -> None:
        """
        初期化する。

        Args:
            tcp_host (str): TCPホスト名
            tcp_port (int): TCPポート番号
            use_grpc (bool): gRPCフラグ
            use_gui (bool): GUIフラグ

        Attributes:
            udi_io (UdiIO): UdiIOインスタンス
            gui_thread (GUIThread | None): GUIスレッド
            command_queue (Queue | None): コマンド受信キュー
        """
        self.udi_io: UdiIO = self._create_udi_io(tcp_host=tcp_host, tcp_port=tcp_port, use_grpc=use_grpc)
        self.gui_thread: GUIThread | None = None
        self.command_queue: Queue | None = None

        # GUIを使用する場合
        if use_gui:
            self.gui_thread = GUIThread()
            self.command_queue = Queue(1)
            self.gui_thread.start(self.command_queue)

    def reset(self) -> StateData:
        """
        デュエルをリセットする。

        Returns:
            StateData: 最初の状態データ
        """
        return self.step(None)

    def step(self, action: ActionData | None) -> StateData:
        """
        デュエルを進める。

        Args:
            action (ActionData | None): 選択された行動データ

        Returns:
            StateData: 次の状態データ
        """
        if action is not None:
            self.udi_io.output_command(action.command_index)

        while True:
            try:
                # UDIから通信が来た場合
                if self.udi_io.input() and self.udi_io.duel_data:
                    # 状態データを取得
                    is_duel_start: bool = self.udi_io.is_duel_start()
                    is_duel_end: bool = self.udi_io.is_duel_end()
                    is_cmd_required: bool = self.udi_io.is_command_required()
                    command_request: CommandRequest = self.udi_io.get_command_request()
                    duel_state_data: DuelStateData = self.udi_io.get_duel_state_data()
                    duel_end_data: DuelEndData | None = self.udi_io.get_duel_end_data() if is_duel_end else None
                    duel_log_data: list[DuelLogDataEntry] = self.udi_io.get_duel_log_data()
                    reward: float = self._compute_reward(duel_end_data)

                    # GUIを更新
                    if self.gui_thread is not None:
                        self.gui_thread.set_data(
                            duel_log_data=duel_log_data,
                            command_request=command_request,
                            duel_state_data=duel_state_data,
                        )

                    # デュエル開始
                    if is_duel_start:
                        print("★★★ Duel Start ★★★")

                    # デュエル終了
                    if is_duel_end and duel_end_data is not None:
                        result_type: ResultType = cast(ResultType, duel_end_data.result_type)
                        finish_type: FinishType = cast(FinishType, duel_end_data.finish_type)
                        print(f"★★★ Duel End ({result_type.name}, {finish_type.name}) ★★★")

                    # デュエル終了 or 行動要求
                    if is_duel_end or is_cmd_required:
                        return StateData(
                            is_duel_start=is_duel_start,
                            is_duel_end=is_duel_end,
                            is_cmd_required=is_cmd_required,
                            command_request=command_request,
                            duel_state_data=duel_state_data,
                            duel_end_data=duel_end_data,
                            duel_log_data=duel_log_data,
                            reward=reward,
                        )

            except Exception as e:
                raise IOError(f"UdiIO failed: {e}") from e

            time.sleep(0.001)

    def _create_udi_io(self, tcp_host: str, tcp_port: int, use_grpc: bool) -> UdiIO:
        """
        UdiIOインスタンスを生成する。

        Args:
            tcp_host (str): TCPホスト名
            tcp_port (int): TCPポート番号
            use_grpc (bool): gRPCフラグ

        Returns:
            UdiIO: UdiIOインスタンス
        """

        connect_type: UdiIO.Connect = UdiIO.Connect.GRPC if use_grpc else UdiIO.Connect.SOCKET

        udi_io: UdiIO = UdiIO(
            tcp_host=tcp_host,
            tcp_port=tcp_port,
            connect=connect_type,
            api_version=1,
        )
        udi_io.log_response_history = False

        return udi_io

    def _compute_reward(self, duel_end_data: DuelEndData | None) -> float:
        """
        報酬を計算する。

        Args:
            duel_end_data (DuelEndData | None): デュエル結果

        Returns:
            float: 報酬
        """
        if duel_end_data is None:
            return 0.0

        result_type: int = duel_end_data.result_type

        if result_type == ResultType.WIN:
            return 1.0
        elif result_type == ResultType.LOSE:
            return -1.0

        return 0.0
