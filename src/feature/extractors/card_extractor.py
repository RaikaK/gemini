import numpy as np

from ygo.constants.enums import Face, PosId, Turn
from ygo.models.duel_card import DuelCard

from src.feature.card_cell_layout import CardCellLayout
from src.feature.card_data import CARD_MAP


class CardExtractor:
    """
    カード特徴量抽出器
    """

    # --- 正規化用定数 ---
    MAX_DECK_COUNT = 40.0
    MAX_CARD_COUNT = 3.0
    MAX_ATK = 3100.0
    MAX_DEF = 2600.0
    MAX_LEVEL = 9.0
    DENOM_SMALL = 5.0
    DENOM_MEDIUM = 10.0
    DENOM_LARGE = 20.0

    # --- チャンネルサイズ ---
    CHANNEL_EMPTY = 0
    SIZE_CARD_ID = 32
    SIZE_ATK = 1
    SIZE_DEF = 1
    SIZE_LEVEL = 1
    SIZE_CATEGORY = 10
    SIZE_POSITION = 2
    SIZE_TURN_PASSED = 1
    SIZE_EFFECT_USED = 3
    SIZE_IS_ATTACKING = 1
    SIZE_IS_ATTACKED = 1
    SIZE_EQUIP_TARGET = 1

    def __init__(self, scaling_factor: float) -> None:
        """
        初期化する。

        Args:
            scaling_factor (float): スケーリング係数

        Attributes:
            scaling_factor (float): スケーリング係数
        """
        self.scaling_factor: float = scaling_factor

    def extract(self, duel_card_table: list[DuelCard], feature: np.ndarray) -> None:
        """
        特徴量を抽出する。

        Args:
            duel_card_table (list[DuelCard]): カード情報リスト
            feature (np.ndarray): 特徴量埋め込み先
        """
        # 初期化
        feature[self.CHANNEL_EMPTY, :, :] = 1.0

        # カウント
        location_counts: dict[tuple[int, int], int] = {}

        for duel_card in duel_card_table:
            key = (duel_card.player_id, duel_card.pos_id)
            location_counts[key] = location_counts.get(key, 0) + 1

        # 埋め込み
        for duel_card in duel_card_table:
            height, width, is_bag = CardCellLayout.get_coord(
                duel_card.player_id, duel_card.pos_id, duel_card.card_index
            )

            if height != -1:
                # 存在情報 (Grid & Bag)
                feature[self.CHANNEL_EMPTY, height, width] = 0.0

                cursor: int = 1

                # カードID情報 (Grid & Bag)
                self._fill_card_id(
                    feature[cursor : cursor + self.SIZE_CARD_ID, :, :],
                    height,
                    width,
                    duel_card,
                    is_bag,
                )
                cursor += self.SIZE_CARD_ID

                # カウント取得
                location_count = location_counts.get((duel_card.player_id, duel_card.pos_id), 0)

                # 攻撃力 (Grid & Bag)
                self._fill_atk(
                    feature[cursor : cursor + self.SIZE_ATK, :, :], height, width, duel_card, location_count, is_bag
                )
                cursor += self.SIZE_ATK

                # 守備力 (Grid & Bag)
                self._fill_def(
                    feature[cursor : cursor + self.SIZE_DEF, :, :], height, width, duel_card, location_count, is_bag
                )
                cursor += self.SIZE_DEF

                # レベル (Grid & Bag)
                self._fill_level(
                    feature[cursor : cursor + self.SIZE_LEVEL, :, :], height, width, duel_card, location_count, is_bag
                )
                cursor += self.SIZE_LEVEL

                # カテゴリ (Grid & Bag)
                self._fill_category(
                    feature[cursor : cursor + self.SIZE_CATEGORY, :, :], height, width, duel_card, is_bag
                )
                cursor += self.SIZE_CATEGORY

                # カードID情報以外 (Grid)
                if not is_bag:
                    # 表示形式
                    self._fill_position(feature[cursor : cursor + self.SIZE_POSITION, :, :], height, width, duel_card)
                    cursor += self.SIZE_POSITION

                    # ターン経過済みフラグ
                    self._fill_turn_passed(
                        feature[cursor : cursor + self.SIZE_TURN_PASSED, :, :], height, width, duel_card
                    )
                    cursor += self.SIZE_TURN_PASSED

                    # 効果使用済みフラグ
                    self._fill_effect_used(
                        feature[cursor : cursor + self.SIZE_EFFECT_USED, :, :], height, width, duel_card
                    )
                    cursor += self.SIZE_EFFECT_USED

                    # 攻撃中フラグ
                    self._fill_is_attacking(
                        feature[cursor : cursor + self.SIZE_IS_ATTACKING, :, :], height, width, duel_card
                    )
                    cursor += self.SIZE_IS_ATTACKING

                    # 攻撃対象フラグ
                    self._fill_is_attacked(
                        feature[cursor : cursor + self.SIZE_IS_ATTACKED, :, :], height, width, duel_card
                    )
                    cursor += self.SIZE_IS_ATTACKED

                    # 装備対象フラグ
                    self._fill_equip_target(
                        feature[cursor : cursor + self.SIZE_EQUIP_TARGET, :, :], duel_card, duel_card_table
                    )
                    cursor += self.SIZE_EQUIP_TARGET

    # =================================================================
    # 埋め込みロジック
    # =================================================================

    def _fill_card_id(self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard, is_bag: bool) -> None:
        """
        カードIDを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
            is_bag (bool): Bagフラグ
        """
        card_id: int = duel_card.card_id
        channel_idx: int = 0

        if card_id != 0 and card_id in CARD_MAP:
            channel_idx = CARD_MAP[card_id]["idx"]

        if is_bag:
            denom: float = self.MAX_DECK_COUNT if channel_idx == 0 else self.MAX_CARD_COUNT
            feature[channel_idx, height, width] += (1.0 / denom) * self.scaling_factor

        else:
            feature[channel_idx, height, width] = 1.0

    def _fill_atk(
        self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard, location_count: int, is_bag: bool
    ) -> None:
        """
        攻撃力を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
            location_count (int): カード総数 (同一位置)
            is_bag (bool): Bagフラグ
        """
        atk: float = 0.0

        if duel_card.pos_id in (PosId.HAND, PosId.GRAVE, PosId.DECK):
            if duel_card.card_id in CARD_MAP:
                atk = float(CARD_MAP[duel_card.card_id]["atk"])

        else:
            atk = float(max(0, duel_card.atk_val))

        if is_bag:
            denom: float = float(location_count) if location_count > 0 else 1.0
            feature[0, height, width] += ((atk / denom) / self.MAX_ATK) * self.scaling_factor

        else:
            feature[0, height, width] = (atk / self.MAX_ATK) * self.scaling_factor

    def _fill_def(
        self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard, location_count: int, is_bag: bool
    ) -> None:
        """
        守備力を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
            location_count (int): カード総数 (同一位置)
            is_bag (bool): Bagフラグ
        """
        def_val: float = 0.0

        if duel_card.pos_id in (PosId.HAND, PosId.GRAVE, PosId.DECK):
            if duel_card.card_id in CARD_MAP:
                def_val = float(CARD_MAP[duel_card.card_id]["def"])

        else:
            def_val = float(max(0, duel_card.def_val))

        if is_bag:
            denom: float = float(location_count) if location_count > 0 else 1.0
            feature[0, height, width] += ((def_val / denom) / self.MAX_DEF) * self.scaling_factor

        else:
            feature[0, height, width] = (def_val / self.MAX_DEF) * self.scaling_factor

    def _fill_level(
        self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard, location_count: int, is_bag: bool
    ) -> None:
        """
        レベルを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
            location_count (int): カード総数 (同一位置)
            is_bag (bool): Bagフラグ
        """
        level_val: float = 0.0

        if duel_card.pos_id in (PosId.HAND, PosId.GRAVE, PosId.DECK):
            if duel_card.card_id in CARD_MAP:
                level_val = float(CARD_MAP[duel_card.card_id]["level"])

        else:
            level_val = float(max(0, duel_card.level))

        if is_bag:
            denom: float = float(location_count) if location_count > 0 else 1.0
            feature[0, height, width] += ((level_val / denom) / self.MAX_LEVEL) * self.scaling_factor

        else:
            feature[0, height, width] = (level_val / self.MAX_LEVEL) * self.scaling_factor

    def _fill_category(self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard, is_bag: bool) -> None:
        """
        カテゴリを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
            is_bag (bool): Bagフラグ
        """
        card_id: int = duel_card.card_id

        def _apply(channel_idx: int, denom: float) -> None:
            if is_bag:
                feature[channel_idx, height, width] += (1.0 / denom) * self.scaling_factor

            else:
                feature[channel_idx, height, width] = 1.0

        if card_id in CARD_MAP:
            card_info: dict[str, int] = CARD_MAP[card_id]

            # モンスター
            if card_info["monster"] == 1:
                _apply(0, self.DENOM_LARGE)

            # 魔法
            if card_info["spell"] == 1:
                _apply(1, self.DENOM_LARGE)

            # 罠
            if card_info["trap"] == 1:
                _apply(2, self.DENOM_MEDIUM)

            # 速攻魔法
            if card_info["quick"] == 1:
                _apply(3, self.DENOM_SMALL)

            # 永続・装備
            if card_info["remains"] == 1:
                _apply(4, self.DENOM_SMALL)

            # 儀式関連
            if card_info["ritual"] == 1:
                _apply(5, self.DENOM_MEDIUM)

            # 通常モンスター
            if card_info["normal"] == 1:
                _apply(6, self.DENOM_MEDIUM)

            # ドラゴン族
            if card_info["dragon"] == 1:
                _apply(7, self.DENOM_LARGE)

            # ATK 1500 以下
            if card_info["monster"] == 1 and card_info["atk"] <= 1500:
                _apply(8, self.DENOM_MEDIUM)

            # レベル 4
            if card_info["monster"] == 1 and card_info["level"] == 4:
                _apply(9, self.DENOM_LARGE)

    def _fill_position(self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard) -> None:
        """
        表示形式を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
        """
        is_field: bool = (PosId.MONSTER_L_L <= duel_card.pos_id <= PosId.MONSTER_R_R) or (
            PosId.MAGIC_L_L <= duel_card.pos_id <= PosId.MAGIC_R_R
        )

        if is_field:
            if duel_card.face == Face.FRONT:
                feature[0, height, width] = 1.0

            is_horizontal: bool = duel_card.turn == Turn.HORIZONTAL
            is_back: bool = duel_card.face == Face.BACK

            if is_horizontal or is_back:
                feature[1, height, width] = 1.0

    def _fill_turn_passed(self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard) -> None:
        """
        ターン経過済みフラグを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
        """
        if duel_card.turn_passed == 1:
            feature[0, height, width] = 1.0

    def _fill_effect_used(self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard) -> None:
        """
        効果使用済みフラグを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
        """
        if duel_card.used_effect1 == 1:
            feature[0, height, width] = 1.0

        if duel_card.used_effect2 == 1:
            feature[1, height, width] = 1.0

        if duel_card.used_effect3 == 1:
            feature[2, height, width] = 1.0

    def _fill_is_attacking(self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard) -> None:
        """
        攻撃中フラグを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
        """
        if duel_card.is_attacking == 1:
            feature[0, height, width] = 1.0

    def _fill_is_attacked(self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard) -> None:
        """
        攻撃対象フラグを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
        """
        if duel_card.is_attacked == 1:
            feature[0, height, width] = 1.0

    def _fill_equip_target(
        self,
        feature: np.ndarray,
        duel_card: DuelCard,
        duel_cards: list[DuelCard],
    ) -> None:
        """
        装備対象を埋め込む。

        Args:
            feature (np.ndarray): 特徴量
            duel_card (DuelCard): カード情報 (装備魔法)
            duel_cards (list[DuelCard]): カード情報リスト (対象モンスター)
        """
        if duel_card.equip_target != -1:
            target_idx: int = duel_card.equip_target

            if 0 <= target_idx < len(duel_cards):
                target_card = duel_cards[target_idx]

                target_height, target_width, _ = CardCellLayout.get_coord(
                    target_card.player_id, target_card.pos_id, target_card.card_index
                )

                if target_height != -1:
                    feature[0, target_height, target_width] = 1.0
