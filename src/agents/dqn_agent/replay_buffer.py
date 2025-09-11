import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import random
from collections import deque

from ygo.constants.enums import SelectionType

from src.ygo_env_wrapper.action_data import ActionData


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state:dict, action_data: ActionData, reward: float, done:bool, next_state: dict):
        """状態sの時の行動ActionData、その後の次状態s’をBufferに追加"""
        replay_data = {
            "state": state, "action_data": action_data, "reward": reward, "done": done, "next_state": next_state
        }
        
        self.buffer.append(replay_data)

        print(f"data_size: {len(self.buffer)}")
    
    def get_batch(self) -> list[dict]|None:
        if len(self.buffer) < self.batch_size:
            return None
        batch_data = random.sample(self.buffer, self.batch_size)
        return batch_data