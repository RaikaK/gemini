import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import torch


class DeepQNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, r_dropout=0.2):
        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(r_dropout),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(r_dropout),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(r_dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(r_dropout),
            torch.nn.Linear(256, output_size)
        )

    def forward(self, input_tensor):
        output_tensor =  self.sequence(input_tensor)
        return output_tensor


def convert_state_to_tensor(state_dict:dict) -> torch.Tensor:
    # DuelStateData

    # CommandRequest
    pass