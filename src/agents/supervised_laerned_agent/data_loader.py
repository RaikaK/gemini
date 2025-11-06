import torch
import numpy as np
import os

from src.config import DEMONSTRATION_DIR
from src.agents.base_agent import BaseAgent
from src.env.action_data import ActionData
from src.env.state_data import StateData
import pickle
from src.common.sample_tensor import (
    set_board_vector,
    set_action_vector,
    create_input_data,
    BOARD_NUM,
    INFO_NUM,
    DNN_INPUT_NUM,
)


class DataLoader:
    def __init__(self, is_each_step: bool = True):
        """
        Args:
            is_each_step (bool): Trueなら各step単位、Falseなら各episode単位でデータを読み込む
        """
        self.buffer: list = (
            self._load_data_each_step()
            if is_each_step
            else self._load_data_each_episode()
        )

    def pop(self) -> tuple[torch.Tensor, torch.Tensor]:
        """X_train, y_trainをbatch_size分取り出す"""

    def _load_data_each_episode(self) -> list[list]:
        """各データの単位がepisode"""
        data_list = []
        for root, _, files in os.walk(DEMONSTRATION_DIR):
            for fname in files:
                if fname.lower().endswith(".pkl"):
                    path = os.path.join(root, fname)
                    try:
                        with open(path, "rb") as f:
                            obj = pickle.load(f)
                        data_list.append(obj)
                    except Exception as e:
                        print(f"Failed to load {path}: {e}")
        # breakpoint()
        return data_list

    def _load_data_each_step(self) -> list:
        """各データの単位がstep"""
        data_list_each_episode = self._load_data_each_episode()
        data_list_each_step = []
        for episode_data in data_list_each_episode:
            data_list_each_step.extend(episode_data)
        return data_list_each_step

    def split_data(self, data_list: list, train_ratio: float) -> tuple[list, list]:
        """
        データリストを2つに分割する
        Returns:
            tuple[list(train | ratio), list(other | 1-ratio)]
        """
        indices = list(range(len(data_list)))
        np.random.shuffle(indices)
        train_size = int(len(data_list) * train_ratio)
        train_indices = indices[:train_size]
        other_indices = indices[train_size:]
        train_data = [data_list[i] for i in train_indices]
        other_data = [data_list[i] for i in other_indices]
        return (train_data, other_data)
