import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import abc

from src.ygo_env_wrapper.action_data import ActionData


class BaseYgoAgent(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def select_action(self, state: dict) -> ActionData:
        """状態stateに基づき、行動データActionDataを返す"""
        pass

    @abc.abstractmethod
    def update(self, action_data: ActionData, next_state: dict):
        """行動データと次状態を用いてエージェントの学習を行う"""
        pass
