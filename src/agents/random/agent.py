import random

from src.agents.base_agent import BaseAgent
from src.env.action_data import ActionData
from src.env.state_data import StateData

from ygo.models.command_request import CommandEntry, CommandRequest


class RandomAgent(BaseAgent):
    """
    ランダムエージェント
    """

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        command_request: CommandRequest = state.command_request
        selectable_commands: list[CommandEntry] = command_request.commands
        selected_command: CommandEntry = random.choice(selectable_commands)
        action = ActionData(command_request=command_request, command_entry=selected_command)

        return action, None

    def update(self, state: StateData, action: ActionData, next_state: StateData, info: dict | None) -> dict | None:
        return None
