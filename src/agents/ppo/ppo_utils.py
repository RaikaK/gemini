import numpy as np
import torch
from torch.distributions import Categorical

from ygo.models.command_request import CommandEntry, CommandRequest
from ygo.models.duel_state_data import DuelStateData
from src.env.state_data import StateData
from src.env.action_data import ActionData

"""
PPOで使用する汎用関数
"""
