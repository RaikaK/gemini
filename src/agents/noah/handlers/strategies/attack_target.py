import random

from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData


def select_attack_target(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """攻撃対象選択 & 値無し"""
    return random.randint(0, len(selectable_commands) - 1)
