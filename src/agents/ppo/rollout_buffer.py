import numpy as np
from ygo.models.command_request import CommandEntry, CommandRequest
from ygo.models.duel_state_data import DuelStateData
from src.env.state_data import StateData
from src.env.action_data import ActionData


class RolloutBuffer:
    def __init__(self):
        self.states: list[StateData] = []
        self.actions: list[ActionData] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.next_states: list[StateData] = []
        self.infos = []

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.infos = []

    def add(
        self,
        state: StateData,
        action: ActionData,
        reward: float,
        done: bool,
        next_state: StateData,
        info: dict,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.infos.append(info)

    def rollout(self) -> dict[list]:
        data = {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "next_states": self.next_states,
            "infos": self.infos,
        }
        self.clear()
        breakpoint()  # dataの中身がからでないか確認
        return data
