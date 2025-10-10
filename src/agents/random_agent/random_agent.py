import sys

sys.path.append("C:/Users/b1/Desktop/u-ni-yo")

import random

from src.agents.base_ygo_agent import BaseYgoAgent
from src.ygo_env_wrapper.action_data import ActionData

from ygo.models import CommandEntry

# シミュレータ起動コマンド
# DuelSimulator.exe --deck_path0 .\DeckData\SimpleBE.json --deck_path1 .\DeckData\SimpleBE.json --randomize_seed true --loop_num 100000 --exit_with_udi true --connect gRPC --tcp_port0 52010 --tcp_port1 52011 --player_type0 Human --player_type1 Human --play_reverse_duel true --grpc_deadline_seconds 60 --log_level 2 --workdir ./workdir1


class RandomAgent(BaseYgoAgent):
    def __init__(self):
        print("RandomAgent")
        return

    def select_action(self, state) -> tuple[ActionData, dict]:
        command_request = state["command_request"]
        selectable_commands: list[CommandEntry] = command_request.commands

        selected_command_entry: CommandEntry = random.choice(selectable_commands)
        action_data = ActionData(
            state=state,
            command_entry=selected_command_entry,
        )
        # print(f"selected cmd index: {action_data.command_index}/[0-{len(action_data.command_request.commands)-1}]")
        return action_data, {}

    def update(
        self, state: dict, action_data: ActionData, next_state: dict, info: dict
    ) -> dict | None:
        # print("RandomAgent does not learn")
        return None
