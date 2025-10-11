import queue
import time

from ygo.gui.udi_gui_thread import UdiGUIThread
from ygo.models.command_request import CommandEntry, CommandRequest

from src.agents.base_agent import BaseAgent
from src.env.action_data import ActionData
from src.env.state_data import StateData


class HumanAgent(BaseAgent):
    """人間"""

    def __init__(self):
        self.queue: queue.Queue = queue.Queue(1)
        self.gui_thread = UdiGUIThread()
        self.gui_thread.start(self.queue)

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        command_request: CommandRequest = state.command_request
        selectable_commands: list[CommandEntry] = command_request.commands

        self.gui_thread.set_data(
            duel_log_data=state.duel_log_data,
            command_request=command_request,
            duel_state_data=state.duel_state_data,
        )

        while True:
            try:
                command_index = self.queue.get(block=False)

            except queue.Empty:
                time.sleep(0.001)
                continue

            command_index = int(command_index)

            if 0 <= command_index < len(selectable_commands):
                break

            time.sleep(0.001)

        selected_command: CommandEntry = selectable_commands[command_index]
        action: ActionData = ActionData(command_request=command_request, command_entry=selected_command)

        return action, None

    def update(self, state: StateData, action: ActionData, next_state: StateData, info: dict | None) -> dict | None:
        return None
