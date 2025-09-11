import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import random
import numpy as np


from src.agents.base_ygo_agent import BaseYgoAgent
from src.ygo_env_wrapper.action_data import ActionData

from ygo.models.duel_state_data import DuelStateData
from ygo.models import CommandEntry, CommandRequest
from ygo.constants.enums import SelectionType


# シミュレータ起動コマンド
# DuelSimulator.exe --deck_path0 .\DeckData\SimpleBE.json --deck_path1 .\DeckData\SimpleBE.json --randomize_seed true --loop_num 100000 --exit_with_udi true --connect gRPC --tcp_port0 52010 --tcp_port1 52011 --player_type0 Human --player_type1 Human --play_reverse_duel true --grpc_deadline_seconds 60 --log_level 2 --workdir ./workdir1

from src.agents.dqn_agent.deep_q_network import DeepQNetwork
from src.agents.dqn_agent.replay_buffer import ReplayBuffer
from src.agents.dqn_agent.simple_tensors.duel_state_data_tensor import (
    simple_duel_state_data_tensor,
)


class DQNAgent(BaseYgoAgent):
    def __init__(
        self, gamma=0.9, lr=1e-5, epsilon=0.2, buffer_size=1e5, batch_size=256
    ):
        print("DQNAgent")
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.dqn = DeepQNetwork(input_size=128, output_size=40)
        self.target_net = DeepQNetwork(input_size=128, output_size=40)
        self.replay_buffer = ReplayBuffer(
            buffer_size=buffer_size, batch_size=batch_size
        )
        return

    def select_action(self, state: dict) -> ActionData:
        # epsilon-greedy方による行動選択
        is_explore = np.random.rand()
        if is_explore <= self.epsilon:
            cmd_request: CommandRequest = state["command_request"]
            selected_cmd_entry: CommandEntry = random.choice(cmd_request.commands)
            action_data = ActionData(
                state=state["state"],
                command_request=cmd_request,
                command_entry=selected_cmd_entry,
            )
            return action_data
        else:
            # DeepNeuralNetworkによる推論
            pass

    def update(self, state: dict, action_data: ActionData, next_state: dict):
        # print("RandomAgent does not learn")
        return
