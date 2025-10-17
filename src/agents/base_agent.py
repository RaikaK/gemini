import abc

from src.env.action_data import ActionData
from src.env.state_data import StateData


class BaseAgent(abc.ABC):
    """
    基底エージェント
    """

    @abc.abstractmethod
    def __init__(self):
        """
        初期化する。
        """

    @abc.abstractmethod
    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        """
        行動を選択する。

        Args:
            state (StateData): 現在の状態データ

        Returns:
            tuple:
                ActionData: 選択した行動データ
                dict | None: 行動選択に関する情報
        """

    @abc.abstractmethod
    def update(self, state: StateData, action: ActionData, next_state: StateData, info: dict | None) -> dict | None:
        """
        内部を更新する。

        Args:
            state (StateData): 現在の状態データ
            action (ActionData): 選択した行動データ
            next_state (StateData): 次の状態データ
            info (dict | None): 行動選択に関する情報

        Returns:
            dict | None: 内部更新に関する情報
        """
