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


class YgoEnv:
    """
    遊戯王のGym環境
    """

    def __init__(self, config: dict):
        """
        初期化する。

        Args:
            config (dict): プレイヤーのUDI接続設定
                例: { 'tcp_host': '10.95.102.79', 'tcp_port': 50000, 'gRPC': True }

        Attributes:
            udi_io (UdiIO): プレイヤーのUDI-IOインスタンス
        """
        self.udi_io: UdiIO = self._create_udi_io(config)

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
                raise IOError(f"UDI-IO failed: {e}") from e

            time.sleep(0.001)

    def _create_udi_io(self, config: dict) -> UdiIO:
        """
        UDI-IOインスタンスを生成する。

        Args:
            config (dict): UDI接続設定
                例: { 'tcp_host': '10.95.102.79', 'tcp_port': 50000, 'gRPC': True }

        Returns:
            UdiIO: UDI-IOインスタンス
        """
        try:
            tcp_host: str = config["tcp_host"]
            tcp_port: int = config["tcp_port"]

        except KeyError as e:
            raise ValueError(f"Missing required key in udi_io config: {e}") from e

        connect_type: UdiIO.Connect = UdiIO.Connect.GRPC if config.get("gRPC") else UdiIO.Connect.SOCKET

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
