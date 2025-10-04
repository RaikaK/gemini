import sys

sys.path.append("C:/Users/b1/Desktop/u-ni-yo")

import random
from collections import deque

from ygo.constants.enums import SelectionType

from src.ygo_env_wrapper.action_data import ActionData
from src.agents.dqn_agent.rewards import compute_life_point_reward, nodeck_reward


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.batch_size = batch_size

    def add(
        self,
        state: dict,
        action_data: ActionData,
        reward: float,
        done: bool,
        next_state: dict,
    ):
        lp_rwd = compute_life_point_reward(state=state, next_state=next_state)
        nodeck_rwd = nodeck_reward(duel_end_data=next_state["duel_end_data"])  # デュエル終了時の報酬
        new_reward = reward + lp_rwd + nodeck_rwd
        """状態sの時の行動ActionData、その後の次状態s’をBufferに追加"""
        replay_data = {
            "state": state,
            "action_data": action_data,
            "reward": new_reward,
            "done": done,
            "next_state": next_state,
        }

        self.buffer.append(replay_data)

        # print(f"data_size: {len(self.buffer)}")

    def len(self):
        return len(self.buffer)

    def get_batch(self) -> list[dict] | None:
        if len(self.buffer) < self.batch_size:
            return None
        batch_data = random.sample(self.buffer, self.batch_size)
        return batch_data

    def clear(self):
        self.buffer.clear()
        # breakpoint()

    # # 報酬を単一の報酬に書き換える
    # def update_all_reward(self):
    #     for data in self.buffer:
    #         lp_rwd = compute_life_point_reward(
    #             state=data["state"], next_state=data["next_state"]
    #         )
    #         nodeck_rwd = nodeck_reward(
    #             duel_end_data=data["next_state"]["duel_end_data"]
    #         )  # デュエル終了時の報酬
    #         data["reward"] += lp_rwd + nodeck_rwd

    def extend(self, memory: list):
        self.buffer.extend(memory)
