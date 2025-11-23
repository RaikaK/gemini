import numpy as np

import src.config as config
from src.env.state_data import StateData
from src.feature.extractors.card import CardExtractor


class FeatureManager:
    """
    特徴量マネージャー
    """

    def __init__(self, scaling_factor: float) -> None:
        """
        初期化する。

        Args:
            scaling_factor (float): スケーリング係数

        Attributes:
            card_extractor (CardExtractor): カード特徴量抽出器
        """
        self.card_extractor: CardExtractor = CardExtractor(scaling_factor)

    def to_feature(self, state: StateData) -> np.ndarray:
        """
        特徴量を返す。

        Args:
            state (StateData): 状態データ

        Returns:
            np.ndarray: 特徴量
        """

        feature: np.ndarray = np.zeros(
            (config.TOTAL_CHANNELS, config.HEIGHT, config.WIDTH),
            dtype=np.float32,
        )

        self.card_extractor.extract(
            state.duel_state_data.duel_card_table,
            feature[0 : config.CHANNELS_CARD, :, :],
        )

        return feature
