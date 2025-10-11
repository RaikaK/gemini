import abc

from src.env.action_data import ActionData
from src.env.state_data import StateData


class BaseAgent(abc.ABC):
    """遊戯王エージェントの基底クラス"""

    @abc.abstractmethod
    def __init__(self):
        """初期化する。"""

    @abc.abstractmethod
    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        """状態stateに基づき、行動データActionDataを返す。"""

    @abc.abstractmethod
    def update(self, state: StateData, action: ActionData, next_state: StateData, info: dict | None) -> dict | None:
        """状態とその時の行動、次状態を取得してエージェントの内部状態を更新する。"""
