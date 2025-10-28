from datetime import datetime
from pathlib import Path
import pickle
from queue import Empty, Queue
import time

from ygo.models.command_request import CommandEntry, CommandRequest

from src.agents.base_agent import BaseAgent
import src.config as config
from src.env.action_data import ActionData
from src.env.state_data import StateData


class HumanAgent(BaseAgent):
    """
    人間エージェント
    """

    def __init__(self, command_queue: Queue, save_demo: bool = False) -> None:
        """
        初期化する。

        Args:
            command_queue (Queue): コマンド受信キュー

        Attributes:
            command_queue (Queue): コマンド受信キュー
            save_demo (bool): デモ保存フラグ
            demo_buffer (list): デモバッファ
        """
        self.command_queue: Queue = command_queue
        self.save_demo: bool = save_demo
        self.demo_buffer: list = []

        if save_demo:
            config.DEMONSTRATION_DIR.mkdir(parents=True, exist_ok=True)

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        command_request: CommandRequest = state.command_request
        selectable_commands: list[CommandEntry] = command_request.commands

        while True:
            try:
                command_index: int = int(self.command_queue.get(block=False))

            except (Empty, ValueError, TypeError):
                time.sleep(0.001)
                continue

            if command_index in range(len(selectable_commands)):
                break

            time.sleep(0.001)

        selected_command: CommandEntry = selectable_commands[command_index]
        action: ActionData = ActionData(command_request=command_request, command_entry=selected_command)

        return action, None

    def update(self, state: StateData, action: ActionData, next_state: StateData, info: dict | None) -> dict | None:
        if not self.save_demo:
            return None

        self.demo_buffer.append(
            {
                "state": state,
                "action": action,
                "next_state": next_state,
                "info": info,
            }
        )

        if next_state.is_duel_end:
            self.save_demonstration()
            self.demo_buffer.clear()

        return None

    def save_demonstration(self) -> None:
        """
        デモを保存する。
        """
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename: Path = config.DEMONSTRATION_DIR / f"{timestamp}.pkl"

        try:
            with open(filename, "wb") as f:
                pickle.dump(self.demo_buffer, f)

        except Exception as e:
            raise IOError(f"Demo saving failed: {e}") from e
