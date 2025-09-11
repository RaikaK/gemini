import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import torch

from ygo.models.command_request import CommandEntry

from src.agents.dqn_agent.simple_tensors.simple_tensor import simple_dataclass_tensor

DIM_COMMAND_ENTRY = len(CommandEntry.__dataclass_fields__)

def simple_command_entry_tenosr(command_entry: CommandEntry) -> torch.Tensor:
    return simple_dataclass_tensor(data=command_entry)