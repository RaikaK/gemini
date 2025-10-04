import sys

sys.path.append("C:/Users/b1/Desktop/u-ni-yo")

from enum import Enum, auto, StrEnum


class EnvStateColumn(StrEnum):
    IsDuelStart = "is_duel_start"
    IsDuelEnd = "is_duel_end"
    IsCommandRequired = "is_cmd_required"
    DuelStateData = "state"
    DuelEndData = "duel_end_data"
    CommandRequest = "command_request"
    Reward = "reward"
