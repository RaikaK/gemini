import numpy as np

from ygo.constants.enums import ChainState, EffectNo
from ygo.models.chain_data import ChainData
from ygo.models.duel_card import DuelCard

from src.feature.card_cell_layout import CardCellLayout


class ChainExtractor:
    """
    チェーン特徴量抽出器
    """

    # --- チャンネルサイズ ---
    SIZE_IS_CHAINING = 1
    SIZE_CHAIN_STATE = 2
    SIZE_CHAIN_SOURCE = 3
    SIZE_CHAIN_TARGET = 3
    SIZE_CHAIN_EFFECT = 3

    def __init__(self, scaling_factor: float) -> None:
        """
        初期化する。

        Args:
            scaling_factor (float): スケーリング係数

        Attributes:
            scaling_factor (float): スケーリング係数
        """
        self.scaling_factor: float = scaling_factor

    def extract(
        self,
        chain_stack: list[ChainData],
        duel_card_table: list[DuelCard],
        feature: np.ndarray,
    ) -> None:
        """
        特徴量を抽出する。

        Args:
            chain_stack (list[ChainData]): チェーン情報リスト
            duel_card_table (list[DuelCard]): カード情報リスト
            feature (np.ndarray): 特徴量埋め込み先
        """
        # 埋め込み
        cursor: int = 0

        # チェーン発生中フラグ
        self._fill_is_chaining(feature[cursor : cursor + self.SIZE_IS_CHAINING, :, :], chain_stack)
        cursor += self.SIZE_IS_CHAINING

        # チェーン状態
        self._fill_chain_state(feature[cursor : cursor + self.SIZE_CHAIN_STATE, :, :], chain_stack)
        cursor += self.SIZE_CHAIN_STATE

        # チェーン発生元
        self._fill_chain_source(
            feature[cursor : cursor + self.SIZE_CHAIN_SOURCE, :, :],
            chain_stack,
            duel_card_table,
        )
        cursor += self.SIZE_CHAIN_SOURCE

        # チェーン対象
        self._fill_chain_target(
            feature[cursor : cursor + self.SIZE_CHAIN_TARGET, :, :],
            chain_stack,
            duel_card_table,
        )
        cursor += self.SIZE_CHAIN_TARGET

        # チェーン効果
        self._fill_chain_effect(
            feature[cursor : cursor + self.SIZE_CHAIN_EFFECT, :, :],
            chain_stack,
            duel_card_table,
        )
        cursor += self.SIZE_CHAIN_EFFECT

    # =================================================================
    # 埋め込みロジック
    # =================================================================

    def _fill_is_chaining(self, feature: np.ndarray, chain_stack: list[ChainData]) -> None:
        """
        チェーン発生中フラグを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            chain_stack (list[ChainData]): チェーン情報リスト
        """
        if len(chain_stack) > 0:
            feature[0, :, :] = 1.0

    def _fill_chain_state(self, feature: np.ndarray, chain_stack: list[ChainData]) -> None:
        """
        チェーン状態を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            chain_stack (list[ChainData]): チェーン情報リスト
        """
        if not chain_stack:
            return

        latest_chain: ChainData = chain_stack[-1]

        if latest_chain.chain_state == ChainState.SET or latest_chain.chain_state == ChainState.WAIT:
            feature[0, :, :] = 1.0

        elif latest_chain.chain_state == ChainState.RESOLVE:
            feature[1, :, :] = 1.0

    def _fill_chain_source(
        self,
        feature: np.ndarray,
        chain_stack: list[ChainData],
        duel_card_table: list[DuelCard],
    ) -> None:
        """
        チェーン発生元を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            chain_stack (list[ChainData]): チェーン情報リスト
            duel_card_table (list[DuelCard]): カード情報リスト
        """
        for chain in chain_stack:
            if 0 <= chain.table_index < len(duel_card_table):
                duel_card: DuelCard = duel_card_table[chain.table_index]
                height, width, is_bag = CardCellLayout.get_coord(
                    duel_card.player_id, duel_card.pos_id, duel_card.card_index
                )

                if height != -1 and not is_bag:
                    channel_idx = min(chain.chain_num, self.SIZE_CHAIN_SOURCE) - 1
                    feature[channel_idx, height, width] = 1.0

    def _fill_chain_target(
        self,
        feature: np.ndarray,
        chain_stack: list[ChainData],
        duel_card_table: list[DuelCard],
    ) -> None:
        """
        チェーン対象を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            chain_stack (list[ChainData]): チェーン情報リスト
            duel_card_table (list[DuelCard]): カード情報リスト
        """
        for chain in chain_stack:
            for target_idx in chain.target_table_index_list:
                if 0 <= target_idx < len(duel_card_table):
                    duel_card: DuelCard = duel_card_table[target_idx]
                    height, width, _ = CardCellLayout.get_coord(
                        duel_card.player_id, duel_card.pos_id, duel_card.card_index
                    )

                    if height != -1:
                        channel_idx = min(chain.chain_num, self.SIZE_CHAIN_TARGET) - 1
                        feature[channel_idx, height, width] = 1.0

    def _fill_chain_effect(
        self,
        feature: np.ndarray,
        chain_stack: list[ChainData],
        duel_card_table: list[DuelCard],
    ) -> None:
        """
        チェーン効果を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            chain_stack (list[ChainData]): チェーン情報リスト
            duel_card_table (list[DuelCard]): カード情報リスト
        """
        for chain in chain_stack:
            if 0 <= chain.table_index < len(duel_card_table):
                duel_card = duel_card_table[chain.table_index]
                height, width, is_bag = CardCellLayout.get_coord(
                    duel_card.player_id, duel_card.pos_id, duel_card.card_index
                )

                if height != -1 and not is_bag:
                    if chain.effect_no == EffectNo.NUM1:
                        feature[0, height, width] = 1.0

                    elif chain.effect_no == EffectNo.NUM2:
                        feature[1, height, width] = 1.0

                    elif chain.effect_no == EffectNo.NUM3:
                        feature[2, height, width] = 1.0
