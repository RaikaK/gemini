import numpy as np

from ygo.models.command_request import CommandEntry

import src.config as config
from src.env.action_data import ActionData
from src.env.state_data import StateData
from src.feature.extractors.card_extractor import CardExtractor
from src.feature.extractors.chain_extractor import ChainExtractor
from src.feature.extractors.entry_extractor import EntryExtractor
from src.feature.extractors.general_extractor import GeneralExtractor
from src.feature.extractors.selection_extractor import SelectionExtractor


class FeatureManager:
    """
    特徴量マネージャー
    """

    def __init__(self, scaling_factor: float = 1.0) -> None:
        """
        初期化する。

        Args:
            scaling_factor (float): スケーリング係数

        Attributes:
            card_extractor (CardExtractor): カード特徴量抽出器
            general_extractor (GeneralExtractor): 局面特徴量抽出器
            chain_extractor (ChainExtractor): チェーン特徴量抽出器
            request_extractor (RequestExtractor): 行動要求特徴量抽出器
            entry_extractor (EntryExtractor): 行動選択特徴量抽出器
        """
        self.card_extractor: CardExtractor = CardExtractor(scaling_factor)
        self.general_extractor: GeneralExtractor = GeneralExtractor(scaling_factor)
        self.chain_extractor: ChainExtractor = ChainExtractor(scaling_factor)
        self.selection_extractor: SelectionExtractor = SelectionExtractor(scaling_factor)
        self.entry_extractor: EntryExtractor = EntryExtractor(scaling_factor)

    def to_snapshot_policy_feature(self, state: StateData) -> np.ndarray:
        """
        単一時点の特徴量を抽出する。 (行動を直接予測するモデル用)

        Args:
            state (StateData): 状態データ

        Returns:
            np.ndarray: 特徴量 (Channels, Height, Width)
        """
        return self._extract_state_feature(state)

    def to_trajectory_policy_feature(
        self, state_action_trajectory: list[tuple[StateData | None, ActionData | None]]
    ) -> np.ndarray:
        """
        時系列の特徴量を抽出する。 (行動を直接予測するモデル用)

        Args:
            state_action_trajectory (list[tuple[StateData | None, ActionData | None]]): 状態データと行動データの軌跡

        Returns:
            np.ndarray: 特徴量 (Sequence, Channels, Height, Width)
        """
        feature_list: list[np.ndarray] = []

        for state, action in state_action_trajectory:
            command_entry: CommandEntry | None = action.command_entry if action else None
            feature: np.ndarray = self._extract_state_action_feature(state, command_entry)
            feature_list.append(feature)

        return np.array(feature_list, dtype=np.float32)

    def to_snapshot_value_feature(self, state: StateData) -> list[np.ndarray]:
        """
        単一時点の特徴量を抽出する。 (行動価値を予測するモデル用)

        Args:
            state (StateData): 状態データ

        Returns:
            list[np.ndarray]: 特徴量リスト [ (Channels, Height, Width), ... ]
        """
        features = []

        for command_entry in state.command_request.commands:
            feature = self._extract_state_action_feature(state, command_entry)
            features.append(feature)

        return features

    def _extract_state_feature(self, state: StateData) -> np.ndarray:
        """
        状態の特徴量を抽出する。

        Args:
            state (StateData): 状態データ

        Returns:
            np.ndarray: 状態の特徴量
        """
        # 初期化
        feature: np.ndarray = np.zeros(
            (config.TOTAL_CHANNELS_STATE, config.HEIGHT, config.WIDTH),
            dtype=np.float32,
        )

        cursor: int = 0

        # カード
        self.card_extractor.extract(
            state.duel_state_data.duel_card_table,
            feature[0 : config.CHANNELS_CARD, :, :],
        )
        cursor += config.CHANNELS_CARD

        # 局面
        self.general_extractor.extract(
            state.duel_state_data.general_data,
            state.duel_state_data.duel_card_table,
            feature[cursor : cursor + config.CHANNELS_GENERAL, :, :],
        )
        cursor += config.CHANNELS_GENERAL

        # チェーン
        self.chain_extractor.extract(
            state.duel_state_data.chain_stack,
            state.duel_state_data.duel_card_table,
            feature[cursor : cursor + config.CHANNELS_CHAIN, :, :],
        )
        cursor += config.CHANNELS_CHAIN

        # 行動要求
        self.selection_extractor.extract(
            state.command_request.selection_type,
            state.command_request.selection_id,
            feature[cursor : cursor + config.CHANNELS_SELECTION, :, :],
        )
        cursor += config.CHANNELS_SELECTION

        # 行動ログ
        for i in range(config.MAX_LOG_HISTORY):
            if i < len(state.command_request.command_log):
                command_log_entry = state.command_request.command_log[i]
                self.selection_extractor.extract(
                    command_log_entry.selection_type,
                    command_log_entry.selection_id,
                    feature[cursor : cursor + config.CHANNELS_SELECTION, :, :],
                )
                self.entry_extractor.extract(
                    command_log_entry.command,
                    feature[cursor + config.CHANNELS_SELECTION : cursor + config.CHANNELS_LOG, :, :],
                )

            cursor += config.CHANNELS_LOG

        # 行動選択
        for command in state.command_request.commands:
            self.entry_extractor.extract(
                command,
                feature[cursor : cursor + config.CHANNELS_ENTRY, :, :],
            )

        cursor += config.CHANNELS_ENTRY

        return feature

    def _extract_action_feature(self, command_entry: CommandEntry) -> np.ndarray:
        """
        行動の特徴量を抽出する。

        Args:
            command_entry (CommandEntry): 行動選択情報

        Returns:
            np.ndarray: 行動の特徴量
        """
        # 初期化
        feature: np.ndarray = np.zeros(
            (config.TOTAL_CHANNELS_ACTION, config.HEIGHT, config.WIDTH),
            dtype=np.float32,
        )

        cursor: int = 0

        # 行動選択
        self.entry_extractor.extract(command_entry, feature[cursor : cursor + config.CHANNELS_ENTRY, :, :])
        cursor += config.CHANNELS_ENTRY

        return feature

    def _extract_state_action_feature(
        self,
        state: StateData | None,
        command_entry: CommandEntry | None,
    ) -> np.ndarray:
        """
        状態＋行動の特徴量を抽出する。

        Args:
            state (StateData | None): 状態データ
            command_entry (CommandEntry | None): 行動選択情報

        Returns:
            np.ndarray: 状態＋行動の特徴量
        """
        # 初期化
        feature: np.ndarray = np.zeros(
            (config.TOTAL_CHANNELS_STATE_ACTION, config.HEIGHT, config.WIDTH),
            dtype=np.float32,
        )

        cursor: int = 0

        # 状態の特徴量
        if state is not None:
            state_feature: np.ndarray = self._extract_state_feature(state)
            feature[0 : config.TOTAL_CHANNELS_STATE, :, :] = state_feature

        cursor += config.TOTAL_CHANNELS_STATE

        # 行動の特徴量
        if command_entry is not None:
            action_feature: np.ndarray = self._extract_action_feature(command_entry)
            feature[cursor : cursor + config.TOTAL_CHANNELS_ACTION, :, :] = action_feature

        cursor += config.TOTAL_CHANNELS_ACTION

        return feature
