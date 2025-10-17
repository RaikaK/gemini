import queue
import threading
import time

from ygo.gui.udi_gui_thread import UdiGUIThread
from ygo.models.command_request import CommandEntry, CommandRequest
from ygo.models.udi_log_data import UdiLogData
from ygo.udi_io import UdiIO

from src.agents.base_agent import BaseAgent
from src.env.action_data import ActionData
from src.env.state_data import StateData


class HumanAgent(BaseAgent):
    """
    人間エージェント
    """

    def __init__(self, udi_io: UdiIO) -> None:
        """
        初期化する。

        Args: udi_io (UdiIO): UDI-IOインスタンス

        Attributes:
            udi_io (UdiIO): UDI-IOインスタンス
            command_queue (queue.Queue): コマンドキュー
            last_log_data (UdiLogData | None): 最新のログデータ
            gui_thread (UdiGUIThread): GUIスレッド
            gui_update_thread (threading.Thread): GUI更新スレッド
        """
        self.udi_io: UdiIO = udi_io
        self.command_queue: queue.Queue = queue.Queue(1)
        self.last_log_data: UdiLogData | None = None

        self.gui_thread: UdiGUIThread = UdiGUIThread()
        self.gui_thread.start(self.command_queue)

        self.gui_update_thread: threading.Thread = threading.Thread(target=self._gui_update_loop, daemon=True)
        self.gui_update_thread.start()

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        command_request: CommandRequest = state.command_request
        selectable_commands: list[CommandEntry] = command_request.commands

        while True:
            try:
                command_index: int = int(self.command_queue.get(block=False))

            except (queue.Empty, ValueError, TypeError):
                time.sleep(0.001)
                continue

            if command_index in range(len(selectable_commands)):
                break

            time.sleep(0.001)

        selected_command: CommandEntry = selectable_commands[command_index]
        action: ActionData = ActionData(command_request=command_request, command_entry=selected_command)

        return action, None

    def update(self, state: StateData, action: ActionData, next_state: StateData, info: dict | None) -> dict | None:
        return None

    def _gui_update_loop(self) -> None:
        """
        GUIを更新し続ける。
        """
        while True:
            try:
                log_data: UdiLogData = UdiLogData(
                    command_request=self.udi_io.get_command_request(),
                    duel_state_data=self.udi_io.get_duel_state_data(),
                    duel_log_data=self.udi_io.get_duel_log_data(),
                    selected_command=-1,
                )

                if log_data != self.last_log_data:
                    self.gui_thread.set_data(
                        duel_log_data=log_data.duel_log_data,
                        command_request=log_data.command_request,
                        duel_state_data=log_data.duel_state_data,
                    )
                    self.last_log_data = log_data

            except Exception:
                pass

            time.sleep(0.1)
