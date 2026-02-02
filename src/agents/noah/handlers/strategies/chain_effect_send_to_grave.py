import random

from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..utils import write_debug_log


def select_chain_effect_send_to_grave(
    _: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """チェーンの効果処理中 (SelectionType:10) & 墓地へ送るカードを選択してください。 (SelectionId:6)"""
    # 最良行動を選択
    best_command_index = 0

    if len(selectable_commands) > 1:
        # ランダム選択 (`table_index`)
        def command_signature(command: CommandEntry):
            return {key: val for key, val in vars(command).items() if key not in ("table_index")}

        if any(
            command_signature(command) != command_signature(selectable_commands[0]) for command in selectable_commands
        ):
            write_debug_log(
                selectable_commands,
                selection_type,
                selection_id,
                f"Ambiguous commands: {selectable_commands}",
            )

        best_command_index = random.choice(range(len(selectable_commands)))

    return best_command_index
