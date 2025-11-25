import numpy as np

from ygo.constants.enums import Face, PosId, Turn
from ygo.models.duel_card import DuelCard

from src.feature.card_cell_layout import CardCellLayout


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

    # --- チャンネルサイズ ---
    CHANNEL_EMPTY = 0
    SIZE_CARD_ID = 32
    SIZE_ATK = 1
    SIZE_DEF = 1
    SIZE_LEVEL = 1
    SIZE_POSITION = 2
    SIZE_TURN_PASSED = 1
    SIZE_EFFECT_USED = 1
    SIZE_IS_ATTACKING = 1
    SIZE_IS_ATTACKED = 1
    SIZE_EQUIP_TARGET = 1

    # --- カードマップ ---
    _CARD_MAP: dict[int, dict[str, int]] = {
        1001: {"idx": 1, "atk": 1300, "def": 2000, "level": 4},  # 洞窟に潜む竜
        1002: {"idx": 2, "atk": 1900, "def": 1600, "level": 4},  # サファイアドラゴン
        1003: {"idx": 3, "atk": 2000, "def": 100, "level": 4},  # アレキサンドライドラゴン
        1004: {"idx": 4, "atk": 3000, "def": 2500, "level": 8},  # 青眼の白龍
        1005: {"idx": 5, "atk": 1800, "def": 1000, "level": 4},  # アサルトワイバーン
        1006: {"idx": 6, "atk": 0, "def": 0, "level": 0},  # 強欲な壺
        1007: {"idx": 7, "atk": 0, "def": 0, "level": 0},  # 大嵐
        1008: {"idx": 8, "atk": 0, "def": 0, "level": 0},  # ライトニング・ボルテックス
        1009: {"idx": 9, "atk": 0, "def": 0, "level": 0},  # 早すぎた埋葬
        1010: {"idx": 10, "atk": 0, "def": 0, "level": 0},  # サイクロン
        1011: {"idx": 11, "atk": 0, "def": 0, "level": 0},  # 収縮
        1012: {"idx": 12, "atk": 0, "def": 0, "level": 0},  # 銀龍の轟咆
        1013: {"idx": 13, "atk": 0, "def": 0, "level": 0},  # 聖なるバリア
        1014: {"idx": 14, "atk": 0, "def": 0, "level": 0},  # 砂塵の大竜巻
        1015: {"idx": 15, "atk": 0, "def": 0, "level": 0},  # 激流葬
        1016: {"idx": 16, "atk": 0, "def": 0, "level": 0},  # 強化蘇生
        1017: {"idx": 17, "atk": 1400, "def": 1100, "level": 3},  # 仮面竜
        1018: {"idx": 18, "atk": 1000, "def": 0, "level": 3},  # ボマー・ドラゴン
        1019: {"idx": 19, "atk": 100, "def": 200, "level": 3},  # コドモドラゴン
        1020: {"idx": 20, "atk": 1400, "def": 1000, "level": 4},  # センジュ・ゴッド
        1021: {"idx": 21, "atk": 1400, "def": 1000, "level": 4},  # ソニックバード
        1022: {"idx": 22, "atk": 1400, "def": 1000, "level": 4},  # マンジュ・ゴッド
        1023: {"idx": 23, "atk": 1800, "def": 600, "level": 4},  # 創世の竜騎士
        1024: {"idx": 24, "atk": 1900, "def": 1200, "level": 4},  # 白竜の聖騎士
        1025: {"idx": 25, "atk": 0, "def": 0, "level": 0},  # 死者蘇生
        1026: {"idx": 26, "atk": 0, "def": 0, "level": 0},  # 白竜降臨
        1027: {"idx": 27, "atk": 0, "def": 0, "level": 0},  # 高等儀式術
        1028: {"idx": 28, "atk": 0, "def": 0, "level": 0},  # 月の書
        1029: {"idx": 29, "atk": 0, "def": 0, "level": 0},  # 禁じられた聖槍
        1030: {"idx": 30, "atk": 0, "def": 0, "level": 0},  # 戦線復帰
        1031: {"idx": 31, "atk": 0, "def": 0, "level": 0},  # リビングデッド
    }

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

                # カードID情報以外 (Grid)
                if not is_bag:
                    # 攻撃力
                    self._fill_atk(feature[cursor : cursor + self.SIZE_ATK, :, :], height, width, duel_card)
                    cursor += self.SIZE_ATK

                    # 守備力
                    self._fill_def(feature[cursor : cursor + self.SIZE_DEF, :, :], height, width, duel_card)
                    cursor += self.SIZE_DEF

                    # レベル
                    self._fill_level(feature[cursor : cursor + self.SIZE_LEVEL, :, :], height, width, duel_card)
                    cursor += self.SIZE_LEVEL

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

        if card_id != 0 and card_id in self._CARD_MAP:
            channel_idx = self._CARD_MAP[card_id]["idx"]

        if is_bag:
            denom: float = self.MAX_DECK_COUNT if channel_idx == 0 else self.MAX_CARD_COUNT
            feature[channel_idx, height, width] += (1.0 / denom) * self.scaling_factor

        else:
            feature[channel_idx, height, width] = 1.0

    def _fill_atk(self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard) -> None:
        """
        攻撃力を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
        """
        atk: float = 0.0

        if duel_card.pos_id == PosId.HAND:
            if duel_card.card_id in self._CARD_MAP:
                atk = float(self._CARD_MAP[duel_card.card_id]["atk"])

        else:
            atk = float(max(0, duel_card.atk_val))

        feature[0, height, width] = (atk / self.MAX_ATK) * self.scaling_factor

    def _fill_def(self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard) -> None:
        """
        守備力を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
        """
        def_val: float = 0.0

        if duel_card.pos_id == PosId.HAND:
            if duel_card.card_id in self._CARD_MAP:
                def_val = float(self._CARD_MAP[duel_card.card_id]["def"])

        else:
            def_val = float(max(0, duel_card.def_val))

        feature[0, height, width] = (def_val / self.MAX_DEF) * self.scaling_factor

    def _fill_level(self, feature: np.ndarray, height: int, width: int, duel_card: DuelCard) -> None:
        """
        レベルを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            height (int): height
            width (int): width
            duel_card (DuelCard): カード情報
        """
        level_val: float = 0.0

        if duel_card.pos_id == PosId.HAND:
            if duel_card.card_id in self._CARD_MAP:
                level_val = float(self._CARD_MAP[duel_card.card_id]["level"])

        else:
            level_val = float(max(0, duel_card.level))

        feature[0, height, width] = (level_val / self.MAX_LEVEL) * self.scaling_factor

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
        if duel_card.used_effect1 == 1 or duel_card.used_effect2 == 1 or duel_card.used_effect3 == 1:
            feature[0, height, width] = 1.0

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
        if duel_card.card_id == 1009 and duel_card.equip_target != -1:
            target_idx: int = duel_card.equip_target

            if 0 <= target_idx < len(duel_cards):
                target_card = duel_cards[target_idx]

                target_height, target_width, _ = CardCellLayout.get_coord(
                    target_card.player_id, target_card.pos_id, target_card.card_index
                )

                if target_height != -1:
                    feature[0, target_height, target_width] = 1.0
