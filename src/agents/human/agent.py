import queue
import time

from ygo.models.command_request import CommandEntry, CommandRequest

from src.agents.base_agent import BaseAgent
from src.env.action_data import ActionData
from src.env.state_data import StateData


class HumanAgent(BaseAgent):
    """
    人間エージェント
    """

    def __init__(self, command_queue: queue.Queue) -> None:
        """
        初期化する。

        Args:
            command_queue (queue.Queue): コマンド受信キュー

        Attributes:
            command_queue (queue.Queue): コマンド受信キュー
        """
        self.command_queue: queue.Queue = command_queue

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
