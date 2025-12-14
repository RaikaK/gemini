import random
from collections import deque

from src.agents.dqn_agent.rewards import compute_life_point_reward, nodeck_reward
from src.env.action_data import ActionData
from src.env.state_data import StateData


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(
        self,
        state: StateData,
        action: ActionData,
        reward: float,
        done: bool,
        next_state: StateData,
        info: dict | None,
    ):
        lp_rwd = compute_life_point_reward(state=state, next_state=next_state)
        nodeck_rwd = nodeck_reward(duel_end_data=next_state.duel_end_data)
        new_reward = reward + lp_rwd + nodeck_rwd
        replay_data = {
            "state": state,
            "action": action,
            "reward": new_reward,
            "done": done,
            "next_state": next_state,
            "info": info,
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

    def extend(self, memory: deque):
        self.buffer.extend(memory)
