import numpy as np

import src.config as config
from src.env.state_data import StateData
from src.feature.extractors.card_extractor import CardExtractor
from src.feature.extractors.chain_extractor import ChainExtractor
from src.feature.extractors.general_extractor import GeneralExtractor
from src.feature.extractors.request_extractor import RequestExtractor


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
            general_extractor (GeneralExtractor): 局面特徴量抽出器
            chain_extractor (ChainExtractor): チェーン特徴量抽出器
            request_extractor (RequestExtractor): 行動要求特徴量抽出器
        """
        self.card_extractor: CardExtractor = CardExtractor(scaling_factor)
        self.general_extractor: GeneralExtractor = GeneralExtractor(scaling_factor)
        self.chain_extractor: ChainExtractor = ChainExtractor(scaling_factor)
        self.request_extractor: RequestExtractor = RequestExtractor(scaling_factor)

    def to_feature(self, state: StateData) -> np.ndarray:
        """
        特徴量を返す。

        Args:
            state (StateData): 状態データ

        Returns:
            np.ndarray: 特徴量
        """
        # 初期化
        feature: np.ndarray = np.zeros(
            (config.TOTAL_CHANNELS, config.HEIGHT, config.WIDTH),
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
        self.request_extractor.extract(
            state.command_request,
            feature[cursor : cursor + config.CHANNELS_REQUEST, :, :],
        )
        cursor += config.CHANNELS_REQUEST

        return feature
