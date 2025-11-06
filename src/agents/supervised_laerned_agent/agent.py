import torch
import numpy as np

from src.agents.base_agent import BaseAgent
from src.env.action_data import ActionData
from src.env.state_data import StateData
from src.common.sample_tensor import (
    set_board_vector,
    set_action_vector,
    create_input_data,
    BOARD_NUM,
    INFO_NUM,
    DNN_INPUT_NUM,
)


class SupervisedLearnedAgent(BaseAgent):
    """教師あり学習済みのモデルをロードして推論のみ行いゲームプレイするエージェント"""

    def __init__(self, model_path: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_size = DNN_INPUT_NUM
        self.model: torch.nn.Module = torch.load(model_path)

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        command_request = state.command_request
        # DuelStateData Tensor
        input_data = create_input_data(state)
        board_vector = set_board_vector(input_data)
        action_vector = set_action_vector(input_data)
        cmd_count = len(state.command_request.commands)

        # 入力テンソル
        x = np.empty((cmd_count, self.input_size), dtype=np.float32)
        for i in range(cmd_count):
            x[i][0 : BOARD_NUM + INFO_NUM] = board_vector
            x[i][BOARD_NUM + INFO_NUM : self.input_size] = action_vector[i]

        input_batch_tensor = torch.tensor(x).to(self.device)
        self.model.eval()
        raw_logits: torch.Tensor = self.model(input_batch_tensor)

        # raw_logitsを(n_cmds, input_size)に変更する
        breakpoint()

        logits = raw_logits.reshape((cmd_count, 1))

        action_probs = torch.softmax(logits, dim=0).detach().cpu().numpy().flatten()

        normalized_action_probs = action_probs / np.sum(action_probs)

        cmd_index = np.random.choice(
            len(command_request.commands), p=normalized_action_probs
        )
        cmd_entry = command_request.commands[cmd_index]
        action = ActionData(command_request=command_request, command_entry=cmd_entry)
        info = {"prob": normalized_action_probs[cmd_index]}
        return action, info

    def update(self, state, action, next_state, info) -> dict | None:
        return info
