import torch
import numpy as np

from src.agents.base_agent import BaseAgent
from src.env.action_data import ActionData
from src.env.state_data import StateData
from src.agents.supervised_laerned_agent.model_loader import load_torch_model
from src.agents.supervised_laerned_agent.data_loader import DataLoader
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
        self.model: torch.nn.Module = load_torch_model(model_path=model_path)
        self.model.to(self.device)

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
        logits: torch.Tensor = self.model(input_batch_tensor).reshape((cmd_count,))

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


if __name__ == "__main__":
    # Debug
    agent = SupervisedLearnedAgent(
        model_path="/Users/fujiyamax/home/labwork/master-duel-ai/u-ni-yo/src/agents/supervised_laerned_agent/trained_models/2025-11-07_14-06-28_epoch1.pth"
    )
    data_loader = DataLoader(is_each_step=True, batch_size=32)
    state = data_loader.test_buffer[0]["state"]

    action = agent.select_action(state)
    # breakpoint()
