import numpy as np

from ygo.constants.enums import DmgStepType, Phase, StepType
from ygo.models.general_data import GeneralData


class GeneralExtractor:
    """
    局面特徴量抽出器
    """

    # --- 正規化用定数 ---
    MAX_LP = 8000.0
    MAX_TURN = 20.0

    # --- チャンネルサイズ ---
    SIZE_LP = 2
    SIZE_MY_TURN = 1
    SIZE_SUMMON = 2
    SIZE_PHASE = 5
    SIZE_STEP = 4
    SIZE_DAMAGE_STEP = 3
    SIZE_TURN_NUM = 1

    def __init__(self, scaling_factor: float) -> None:
        """
        初期化する。

        Args:
            scaling_factor (float): スケーリング係数

        Attributes:
            scaling_factor (float): スケーリング係数
        """
        self.scaling_factor: float = scaling_factor

    def extract(self, general_data: GeneralData, feature: np.ndarray) -> None:
        """
        特徴量を抽出する。

        Args:
            general_data (GeneralData): 局面情報
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

        # 召喚権フラグ
        self._fill_summon(feature[cursor : cursor + self.SIZE_SUMMON, :, :], general_data)
        cursor += self.SIZE_SUMMON

        # フェイズフラグ
        self._fill_phase(feature[cursor : cursor + self.SIZE_PHASE, :, :], general_data)
        cursor += self.SIZE_PHASE

        # ステップフラグ
        self._fill_step(feature[cursor : cursor + self.SIZE_STEP, :, :], general_data)
        cursor += self.SIZE_STEP

        # ダメージステップフラグ
        self._fill_damage_step(feature[cursor : cursor + self.SIZE_DAMAGE_STEP, :, :], general_data)
        cursor += self.SIZE_DAMAGE_STEP

        # ターン数
        self._fill_turn_num(feature[cursor : cursor + self.SIZE_TURN_NUM, :, :], general_data)
        cursor += self.SIZE_TURN_NUM

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
        my_lp: float = float(min(general_data.lp[0], self.MAX_LP))
        op_lp: float = float(min(general_data.lp[1], self.MAX_LP))

        feature[0, :, :] = (my_lp / self.MAX_LP) * self.scaling_factor
        feature[1, :, :] = (op_lp / self.MAX_LP) * self.scaling_factor

    def _fill_my_turn(self, feature: np.ndarray, general_data: GeneralData) -> None:
        """
        自分のターンフラグを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            general_data (GeneralData): 局面情報
        """
        if general_data.which_turn_now == 0:
            feature[0, :, :] = 1.0

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

    def _fill_turn_num(self, feature: np.ndarray, general_data: GeneralData) -> None:
        """
        ターン数を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            general_data (GeneralData): 局面情報
        """
        turn_val: float = float(min(general_data.turn_num + 1, self.MAX_TURN))

        feature[0, :, :] = (turn_val / self.MAX_TURN) * self.scaling_factor
