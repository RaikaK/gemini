import torch
import numpy as np
import os
import copy


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
    def __init__(
        self,
        is_each_step: bool = True,
        batch_size: int = 32,
        train_data_ratio: float = 0.8,
    ):
        """
        Args:
            is_each_step (bool): Trueなら各step単位、Falseなら各episode単位でデータを読み込む
        """
        self._is_each_step = is_each_step
        self._train_data_ratio = train_data_ratio
        self._init_train_buffer, self._init_test_buffer = self._split_data(self._load_data(), self._train_data_ratio)
        self.batch_size = batch_size
        self.train_buffer = []
        self.test_buffer = []
        # 実際に使用するデータの用意
        self.reset()

    def get_test_data(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        Xs = []
        ys = []
        for data in self.test_buffer:
            state: StateData = data["state"]
            action: ActionData = data["action"]
            cmd_count = len(action.command_request.commands)
            input_data = create_input_data(state)
            x = np.empty((cmd_count, DNN_INPUT_NUM), dtype=np.float32)
            y = np.zeros((cmd_count,), dtype=np.float32)
            # set x
            for i in range(cmd_count):
                x[i][0 : BOARD_NUM + INFO_NUM] = set_board_vector(input_data)
                x[i][BOARD_NUM + INFO_NUM : DNN_INPUT_NUM] = set_action_vector(input_data)[i]
            # set y
            y[action.command_index] = 1.0
            Xs.append(x)
            ys.append(y)
        return Xs, ys

    def get_train_batch_data(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[int]] | None:
        """
        Returns:
            バッチデータ数のX_batch, y_batch, has_batchを返す
            - X_batch: list[DNNへの入力テンソル]
            - y_batch: list[正解ラベルテンソル]
            - hash_batch: X_batch,y_batchでコマンド数
            - **_train_bufferが空ならば、Noneを返す**
        """
        if len(self.train_buffer) <= 0:
            return None
        batch_data: list[dict] = self.train_buffer[: self.batch_size]
        self.train_buffer = self.train_buffer[self.batch_size :]

        # batch_tensorを作成
        X_batch = []
        y_batch = []
        hash_batch = []
        for data in batch_data:
            state: StateData = data["state"]
            action: ActionData = data["action"]
            cmd_count = len(action.command_request.commands)
            input_data = create_input_data(state)
            x = np.empty((cmd_count, DNN_INPUT_NUM), dtype=np.float32)
            y = np.zeros((cmd_count,), dtype=np.float32)
            # set x
            for i in range(cmd_count):
                x[i][0 : BOARD_NUM + INFO_NUM] = set_board_vector(input_data)
                x[i][BOARD_NUM + INFO_NUM : DNN_INPUT_NUM] = set_action_vector(input_data)[i]
            # set y
            y[action.command_index] = 1.0
            X_batch.append(x)
            y_batch.append(y)
            hash_batch.append(cmd_count)

        return X_batch, y_batch, hash_batch

    def reset(self):
        """
        - データローダーのリセット (get_train_batch_data()でNoneが返されたら、呼び出せば良い)
        - ただし、new()はしないこと -> new()すると、trainデータとテストデータが更新されるため
        - new()するときは、train/testデータを更新したいときのみ行う
        """
        self.train_buffer.clear()
        self.test_buffer.clear()
        self.train_buffer = copy.deepcopy(self._init_train_buffer)
        self.test_buffer = copy.deepcopy(self._init_test_buffer)

    def _load_data(self) -> list:
        """データの読み込み"""
        data_list = self._load_data_each_step() if self._is_each_step else self._load_data_each_episode()
        np.random.shuffle(data_list)
        return data_list

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

    def _split_data(self, data_list: list, train_ratio: float) -> tuple[list, list]:
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


if __name__ == "__main__":
    data_loader = DataLoader()
    for i in range(40):
        print(i)
        batch = data_loader.get_train_batch_data()
