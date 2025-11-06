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


def load_data_each_episode() -> list[list]:
    """各データの単位がepisode"""
    # DEMONSTRATION_DIRから学習データを読み込む
    # DEMONSTRATION_DIRに含まれる.pklファイルを全て読み込む
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


def load_data_each_step() -> list:
    """各データの単位がstep"""
    data_list_each_episode = load_data_each_episode()
    data_list_each_step = []
    for episode_data in data_list_each_episode:
        data_list_each_step.extend(episode_data)
    return data_list_each_step


if __name__ == "__main__":
    data_list = load_data_each_step()
