import numpy as np
from ygo.models.command_request import CommandEntry, CommandRequest
from ygo.models.duel_state_data import DuelStateData
from src.env.state_data import StateData
from src.env.action_data import ActionData
from src.common.sample_rewards import life_point_reward, nodeck_reward


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
        # 報酬は、LPとFinishTypeによって分ける
        new_reward = (
            reward + life_point_reward(state=state, next_state=next_state) + nodeck_reward(next_state.duel_end_data)
        )
        self.rewards.append(new_reward)
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
        # breakpoint()  # dataの中身がからでないか確認
        return data
