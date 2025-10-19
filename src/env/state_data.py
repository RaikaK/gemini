from dataclasses import dataclass

from ygo.models import DuelEndData
from ygo.models.command_request import CommandRequest
from ygo.models.duel_log_data_entry import DuelLogDataEntry
from ygo.models.duel_state_data import DuelStateData


@dataclass
class StateData:
    """
    状態データ
    """

    is_duel_start: bool
    """デュエル開始フラグ"""

    is_duel_end: bool
    """デュエル終了フラグ"""

    is_cmd_required: bool
    """行動要求フラグ"""

    command_request: CommandRequest
    """行動要求"""

    duel_state_data: DuelStateData
    """デュエル状態"""

    duel_end_data: DuelEndData | None
    """デュエル結果"""

    duel_log_data: list[DuelLogDataEntry]
    """デュエルログ"""

    reward: float
    """報酬"""
