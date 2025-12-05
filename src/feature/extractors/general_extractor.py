import numpy as np

from ygo.constants.enums import DmgStepType, Phase, PlayerId, PosId, StepType
from ygo.models.duel_card import DuelCard
from ygo.models.general_data import GeneralData


class GeneralExtractor:
    """
    局面特徴量抽出器
    """

    # --- 正規化用定数 ---
    MAX_LP = 8000.0
    MAX_TURN = 20.0
    COST_PREMATURE_BURIAL = 800.0
    MAX_HAND = 8.0
    MAX_DECK = 35.0
    MAX_GRAVE = 25.0
    MAX_FIELD = 5.0

    # --- チャンネルサイズ ---
    SIZE_LP = 5
    SIZE_MY_TURN = 1
    SIZE_TURN_NUM = 1
    SIZE_TURN_INFO = 2
    SIZE_SUMMON = 2
    SIZE_CARD_NUM = 10
    SIZE_PHASE = 5
    SIZE_STEP = 4
    SIZE_DAMAGE_STEP = 3

    def __init__(self, scaling_factor: float) -> None:
        """
        初期化する。

        Args:
            scaling_factor (float): スケーリング係数

        Attributes:
            scaling_factor (float): スケーリング係数
        """
        self.scaling_factor: float = scaling_factor

    def extract(self, general_data: GeneralData, duel_card_table: list[DuelCard], feature: np.ndarray) -> None:
        """
        特徴量を抽出する。

        Args:
            general_data (GeneralData): 局面情報
            duel_card_table (list[DuelCard]): カード情報リスト
            feature (np.ndarray): 特徴量埋め込み先
        """
        # 埋め込み
        cursor: int = 0

        # LP
        self._fill_lp(feature[cursor : cursor + self.SIZE_LP, :, :], general_data)
        cursor += self.SIZE_LP

        # 自分のターンフラグ
        self._fill_my_turn(feature[cursor : cursor + self.SIZE_MY_TURN, :, :], general_data)
        cursor += self.SIZE_MY_TURN

        # ターン数
        self._fill_turn_num(feature[cursor : cursor + self.SIZE_TURN_NUM, :, :], general_data)
        cursor += self.SIZE_TURN_NUM

        # ターン情報
        self._fill_turn_info(feature[cursor : cursor + self.SIZE_TURN_INFO, :, :], general_data)
        cursor += self.SIZE_TURN_INFO

        # 召喚権フラグ
        self._fill_summon(feature[cursor : cursor + self.SIZE_SUMMON, :, :], general_data)
        cursor += self.SIZE_SUMMON

        # カード枚数
        self._fill_card_num(feature[cursor : cursor + self.SIZE_CARD_NUM, :, :], duel_card_table)
        cursor += self.SIZE_CARD_NUM

        # フェイズフラグ
        self._fill_phase(feature[cursor : cursor + self.SIZE_PHASE, :, :], general_data)
        cursor += self.SIZE_PHASE

        # ステップフラグ
        self._fill_step(feature[cursor : cursor + self.SIZE_STEP, :, :], general_data)
        cursor += self.SIZE_STEP

        # ダメージステップフラグ
        self._fill_damage_step(feature[cursor : cursor + self.SIZE_DAMAGE_STEP, :, :], general_data)
        cursor += self.SIZE_DAMAGE_STEP

    # =================================================================
    # 埋め込みロジック
    # =================================================================

    def _fill_lp(self, feature: np.ndarray, general_data: GeneralData) -> None:
        """
        LPを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            general_data (GeneralData): 局面情報
        """
        my_lp_raw: float = float(general_data.lp[0])
        op_lp_raw: float = float(general_data.lp[1])

        # 自分のLP
        feature[0, :, :] = (min(my_lp_raw, self.MAX_LP) / self.MAX_LP) * self.scaling_factor

        # 相手のLP
        feature[1, :, :] = (min(op_lp_raw, self.MAX_LP) / self.MAX_LP) * self.scaling_factor

        # LP差分
        diff = (my_lp_raw - op_lp_raw) / self.MAX_LP
        feature[2, :, :] = max(0.0, diff) * self.scaling_factor
        feature[3, :, :] = max(0.0, -diff) * self.scaling_factor

        # コスト：早すぎる埋葬
        if my_lp_raw >= self.COST_PREMATURE_BURIAL:
            feature[4, :, :] = 1.0

    def _fill_my_turn(self, feature: np.ndarray, general_data: GeneralData) -> None:
        """
        自分のターンフラグを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            general_data (GeneralData): 局面情報
        """
        if general_data.which_turn_now == 0:
            feature[0, :, :] = 1.0

    def _fill_turn_num(self, feature: np.ndarray, general_data: GeneralData) -> None:
        """
        ターン数を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            general_data (GeneralData): 局面情報
        """
        turn_val: float = float(min(general_data.turn_num + 1, self.MAX_TURN))

        feature[0, :, :] = (turn_val / self.MAX_TURN) * self.scaling_factor

    def _fill_turn_info(self, feature: np.ndarray, general_data: GeneralData) -> None:
        """
        ターン情報を埋め込む。
        Args:
            feature (np.ndarray): 特徴量埋め込み先
            general_data (GeneralData): 局面情報
        """
        # 最初のターンフラグ
        if general_data.turn_num == 0:
            feature[0, :, :] = 1.0

        # 先攻フラグ
        is_even_turn = general_data.turn_num % 2 == 0
        is_my_turn = general_data.which_turn_now == 0

        if is_even_turn == is_my_turn:
            feature[1, :, :] = 1.0

    def _fill_summon(self, feature: np.ndarray, general_data: GeneralData) -> None:
        """
        召喚権フラグを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            general_data (GeneralData): 局面情報
        """
        if general_data.summon_num[0] > 0:
            feature[0, :, :] = 1.0

        if general_data.summon_num[1] > 0:
            feature[1, :, :] = 1.0

    def _fill_card_num(self, feature: np.ndarray, duel_card_table: list[DuelCard]) -> None:
        """
        カード枚数を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            duel_card_table (list[DuelCard]): カード情報リスト
        """
        # カウント
        location_counts: dict[tuple[int, int], int] = {}

        for duel_card in duel_card_table:
            key = (duel_card.player_id, duel_card.pos_id)
            location_counts[key] = location_counts.get(key, 0) + 1

        # 自分手札
        feature[0, :, :] = (
            min(float(location_counts.get((PlayerId.MYSELF, PosId.HAND), 0)), self.MAX_HAND) / self.MAX_HAND
        ) * self.scaling_factor
        # 相手手札
        feature[1, :, :] = (
            min(float(location_counts.get((PlayerId.RIVAL, PosId.HAND), 0)), self.MAX_HAND) / self.MAX_HAND
        ) * self.scaling_factor

        # 自分デッキ
        feature[2, :, :] = (
            min(float(location_counts.get((PlayerId.MYSELF, PosId.DECK), 0)), self.MAX_DECK) / self.MAX_DECK
        ) * self.scaling_factor
        # 相手デッキ
        feature[3, :, :] = (
            min(float(location_counts.get((PlayerId.RIVAL, PosId.DECK), 0)), self.MAX_DECK) / self.MAX_DECK
        ) * self.scaling_factor

        # 自分墓地
        feature[4, :, :] = (
            min(float(location_counts.get((PlayerId.MYSELF, PosId.GRAVE), 0)), self.MAX_GRAVE) / self.MAX_GRAVE
        ) * self.scaling_factor
        # 相手墓地
        feature[5, :, :] = (
            min(float(location_counts.get((PlayerId.RIVAL, PosId.GRAVE), 0)), self.MAX_GRAVE) / self.MAX_GRAVE
        ) * self.scaling_factor

        # 自分モンスター
        monster_num = (
            location_counts.get((PlayerId.MYSELF, PosId.MONSTER_L_L), 0)
            + location_counts.get((PlayerId.MYSELF, PosId.MONSTER_L), 0)
            + location_counts.get((PlayerId.MYSELF, PosId.MONSTER_C), 0)
            + location_counts.get((PlayerId.MYSELF, PosId.MONSTER_R), 0)
            + location_counts.get((PlayerId.MYSELF, PosId.MONSTER_R_R), 0)
        )
        feature[6, :, :] = (min(float(monster_num), self.MAX_FIELD) / self.MAX_FIELD) * self.scaling_factor

        # 相手モンスター
        monster_num = (
            location_counts.get((PlayerId.RIVAL, PosId.MONSTER_L_L), 0)
            + location_counts.get((PlayerId.RIVAL, PosId.MONSTER_L), 0)
            + location_counts.get((PlayerId.RIVAL, PosId.MONSTER_C), 0)
            + location_counts.get((PlayerId.RIVAL, PosId.MONSTER_R), 0)
            + location_counts.get((PlayerId.RIVAL, PosId.MONSTER_R_R), 0)
        )
        feature[7, :, :] = (min(float(monster_num), self.MAX_FIELD) / self.MAX_FIELD) * self.scaling_factor

        # 自分魔法・罠
        magic_trap_num = (
            location_counts.get((PlayerId.MYSELF, PosId.MAGIC_L_L), 0)
            + location_counts.get((PlayerId.MYSELF, PosId.MAGIC_L), 0)
            + location_counts.get((PlayerId.MYSELF, PosId.MAGIC_C), 0)
            + location_counts.get((PlayerId.MYSELF, PosId.MAGIC_R), 0)
            + location_counts.get((PlayerId.MYSELF, PosId.MAGIC_R_R), 0)
        )
        feature[8, :, :] = (min(float(magic_trap_num), self.MAX_FIELD) / self.MAX_FIELD) * self.scaling_factor

        # 相手魔法・罠
        magic_trap_num = (
            location_counts.get((PlayerId.RIVAL, PosId.MAGIC_L_L), 0)
            + location_counts.get((PlayerId.RIVAL, PosId.MAGIC_L), 0)
            + location_counts.get((PlayerId.RIVAL, PosId.MAGIC_C), 0)
            + location_counts.get((PlayerId.RIVAL, PosId.MAGIC_R), 0)
            + location_counts.get((PlayerId.RIVAL, PosId.MAGIC_R_R), 0)
        )
        feature[9, :, :] = (min(float(magic_trap_num), self.MAX_FIELD) / self.MAX_FIELD) * self.scaling_factor

    def _fill_phase(self, feature: np.ndarray, general_data: GeneralData) -> None:
        """
        フェイズフラグを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            general_data (GeneralData): 局面情報
        """
        current_phase: int = general_data.current_phase

        if current_phase == Phase.DRAW or current_phase == Phase.STANDBY:
            feature[0, :, :] = 1.0

        elif current_phase == Phase.MAIN1:
            feature[1, :, :] = 1.0

        elif current_phase == Phase.BATTLE:
            feature[2, :, :] = 1.0

        elif current_phase == Phase.MAIN2:
            feature[3, :, :] = 1.0

        elif current_phase == Phase.END:
            feature[4, :, :] = 1.0

    def _fill_step(self, feature: np.ndarray, general_data: GeneralData) -> None:
        """
        ステップフラグを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            general_data (GeneralData): 局面情報
        """

        current_step: int = general_data.current_step

        if current_step == StepType.START:
            feature[0, :, :] = 1.0

        elif current_step == StepType.BATTLE:
            feature[1, :, :] = 1.0

        elif current_step == StepType.DAMAGE:
            feature[2, :, :] = 1.0

        elif current_step == StepType.END:
            feature[3, :, :] = 1.0

    def _fill_damage_step(self, feature: np.ndarray, general_data: GeneralData) -> None:
        """
        ダメージステップフラグを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            general_data (GeneralData): 局面情報
        """
        current_damage_step: int = general_data.current_damage_step

        if current_damage_step == DmgStepType.START or current_damage_step == DmgStepType.BEFORE_CALC:
            feature[0, :, :] = 1.0

        elif current_damage_step == DmgStepType.DAMAGE_CALC or current_damage_step == DmgStepType.AFTER_CALC:
            feature[1, :, :] = 1.0

        elif current_damage_step == DmgStepType.END:
            feature[2, :, :] = 1.0
