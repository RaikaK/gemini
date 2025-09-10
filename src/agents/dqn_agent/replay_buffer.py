import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

import random
from collections import deque

from src.ygo_env_wrapper.action_data import ActionData

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, data:dict[str, dict|ActionData|dict]):
        """状態sの時の行動ActionData、その後の次状態s’をBufferに追加"""
        self.buffer.append(data)
    
    def get_batch(self):
        if len(self.buffer) <= self.batch_size:
            return self.buffer
        batch_data = random.sample(self.buffer, self.batch_size)
        return batch_data