from pathlib import Path
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from ygo.constants.enums import CommandType
from ygo.models.command_request import CommandEntry, CommandRequest


from src.agents.hierarchical.models.action_heads import ActionHeads
import src.config as config
from src.env.action_data import ActionData
from src.env.state_data import StateData
from src.feature.feature_manager import FeatureManager


class HierarchicalDataset(Dataset):
    """
    階層型データセット
    """

    def __init__(
        self,
        state_features: list[np.ndarray],
        action_labels: list[dict[str, int]],
        command_requests: list[CommandRequest],
        correct_indices: list[int],
    ) -> None:
        """
        初期化する。

        Args:
            state_features (list[np.ndarray]): 状態特徴量リスト
            action_labels (list[dict[str, int]]): 正解行動ラベル（各Headのインデックス）リスト
            command_requests (list[CommandRequest]): コマンドリクエストのリスト
            correct_indices (list[int]): 正解インデックスリスト

        Attributes:
            state_features (list[np.ndarray]): 状態特徴量リスト
            action_labels (list[dict[str, int]]): 正解行動ラベル（各Headのインデックス）リスト
            command_requests (list[CommandRequest]): コマンドリクエストのリスト
            correct_indices (list[int]): 正解インデックスリスト
        """
        self.state_features: list[np.ndarray] = state_features
        self.action_labels: list[dict[str, int]] = action_labels
        self.command_requests: list[CommandRequest] = command_requests
        self.correct_indices: list[int] = correct_indices

    def __len__(self) -> int:
        """
        データ数を返す。

        Returns:
            int: データ数
        """
        return len(self.state_features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], CommandRequest, int]:
        """
        指定されたインデックスのデータを返す。

        Args:
            idx (int): インデックス

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: (状態特徴量, 正解行動ラベル)
        """
        # 状態特徴量をTensorに変換
        state_feature = torch.from_numpy(self.state_features[idx])

        # 正解行動ラベルをTensorに変換
        action_label = {key: torch.tensor(val, dtype=torch.long) for key, val in self.action_labels[idx].items()}

        return state_feature, action_label, self.command_requests[idx], self.correct_indices[idx]


def _process_single_file(
    file_path: Path, feature_manager: FeatureManager
) -> tuple[list[np.ndarray], list[dict[str, int]], list[CommandRequest], list[int]]:
    """
    単一のpklファイルを読み込み、状態特徴量と正解行動ラベルを作成する。

    Args:
        file_path (Path): pklファイルのパス
        feature_manager (FeatureManager): 特徴量マネージャー

    Returns:
        tuple[list[np.ndarray], list[dict[str, int]], list[CommandRequest], list[int]]: (状態特徴量リスト, 正解行動ラベルリスト, コマンドリクエストリスト, 正解インデックスリスト)
    """
    state_features: list[np.ndarray] = []
    action_labels: list[dict[str, int]] = []
    command_requests: list[CommandRequest] = []
    correct_indices: list[int] = []

    try:
        # pklファイルを読み込む
        with open(file_path, "rb") as f:
            demonstration_data: list[dict] = pickle.load(f)

        # 各ステップを処理
        for step_data in demonstration_data:
            state: StateData = step_data["state"]
            action: ActionData = step_data["action"]

            # 選択された行動を取得
            selected_command: CommandEntry = action.command_entry

            # ドローはスキップ
            if selected_command.command_type == CommandType.DRAW:
                continue

            # 状態特徴量を作成
            feature: np.ndarray = feature_manager.to_state_feature(state)

            # 正解行動ラベルを作成
            label_dict: dict[str, int] = {}

            for action_head in ActionHeads.get_all_heads():
                val: int = getattr(selected_command, action_head.name)
                label_dict[action_head.name] = ActionHeads.to_head_index(action_head.name, val)

            state_features.append(feature)
            action_labels.append(label_dict)
            command_requests.append(action.command_request)
            correct_indices.append(action.get_command_index())

    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return [], [], [], []

    return state_features, action_labels, command_requests, correct_indices


def create_hierarchical_datasets(
    feature_manager: FeatureManager, valid_ratio: float = 0.1
) -> tuple[HierarchicalDataset, HierarchicalDataset]:
    """
    学習用と検証用の階層型データセットを作成する。

    Args:
        feature_manager (FeatureManager): 特徴量マネージャー
        valid_ratio (float): 検証データの割合

    Returns:
        tuple[HierarchicalDataset, HierarchicalDataset]: (学習用データセット, 検証用データセット)
    """
    # pklファイルを取得
    pkl_files: list[Path] = list(config.DEMONSTRATION_DIR.glob("*.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {config.DEMONSTRATION_DIR}")

    all_state_features: list[np.ndarray] = []
    all_action_labels: list[dict[str, int]] = []
    all_command_requests: list[CommandRequest] = []
    all_correct_indices: list[int] = []

    # データを作成
    for pkl_file in tqdm(pkl_files, desc="Processing files"):
        state_features, action_labels, command_requests, correct_indices = _process_single_file(
            pkl_file, feature_manager
        )
        all_state_features.extend(state_features)
        all_action_labels.extend(action_labels)
        all_command_requests.extend(command_requests)
        all_correct_indices.extend(correct_indices)

    # シャッフルして分割
    total_samples = len(all_state_features)
    indices = list(range(total_samples))
    random.shuffle(indices)
    valid_size = int(total_samples * valid_ratio)
    train_indices = indices[valid_size:]
    valid_indices = indices[:valid_size]

    # データセットを作成
    train_dataset = HierarchicalDataset(
        [all_state_features[i] for i in train_indices],
        [all_action_labels[i] for i in train_indices],
        [all_command_requests[i] for i in train_indices],
        [all_correct_indices[i] for i in train_indices],
    )
    valid_dataset = HierarchicalDataset(
        [all_state_features[i] for i in valid_indices],
        [all_action_labels[i] for i in valid_indices],
        [all_command_requests[i] for i in valid_indices],
        [all_correct_indices[i] for i in valid_indices],
    )

    return train_dataset, valid_dataset
