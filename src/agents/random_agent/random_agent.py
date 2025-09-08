import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import random

from src.agents.base_ygo_agent import BaseYgoAgent
from src.ygo_env_wrapper.action_data import ActionData

from ygo.models import CommandEntry


class RandomAgent(BaseYgoAgent):
    def __init__(self):
        print("RandomAgent")
        return

    def select_action(self, state) -> ActionData:
        command_request = state["command_request"]
        selectable_commands: list[CommandEntry] = command_request.commands
        duel_state = state["state"]
        is_cmd_required: bool = state["is_cmd_required"]
        if is_cmd_required and len(selectable_commands) > 0:
            selected_command_entry: CommandEntry = random.choice(selectable_commands)
            action_data = ActionData(
                state=duel_state,
                command_request=command_request,
                command_entry=selected_command_entry,
            )
            return action_data

        print("No command selected")
        return None

    def update(self, action_data: ActionData, reward: float, next_state: dict):
        print("RandomAgent does not learn")
        return
