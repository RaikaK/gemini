import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

from ygo.util.text import TextUtil

from src.ygo_env_wrapper.action_data import ActionData
from src.agents.base_ygo_agent import BaseYgoAgent


class LLMAgent(BaseYgoAgent):
    def __init__(self):
        super().__init__()

    def select_action(self, state: dict) -> ActionData:
        """状態stateに基づき、行動データActionDataを返す"""
        # LLMを用いて行動を選択するロジックをここに実装
        pass

    def update(self, state, action_data, next_state) -> dict | None:
        return super().update(state, action_data, next_state)
