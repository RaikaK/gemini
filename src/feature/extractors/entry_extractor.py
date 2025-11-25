import numpy as np

from ygo.constants.enums import CommandType, EffectNo, Phase, Turn, YesNo
from ygo.models.command_request import CommandEntry

from src.feature.card_cell_layout import CardCellLayout


class EntryExtractor:
    """
    行動選択特徴量抽出器
    """

    # --- チャンネルサイズ ---
    SIZE_COMMAND_TYPE = 11
    SIZE_CARD_ID = 32
    SIZE_TARGET_GRID = 1
    SIZE_EFFECT_NO = 3
    SIZE_PHASE = 3
    SIZE_STAND_TURN = 2
    SIZE_YES_NO = 2

    # --- カードマップ ---
    _CARD_ID_MAP: dict[int, int] = {
        1001: 1,
        1002: 2,
        1003: 3,
        1004: 4,
        1005: 5,
        1006: 6,
        1007: 7,
        1008: 8,
        1009: 9,
        1010: 10,
        1011: 11,
        1012: 12,
        1013: 13,
        1014: 14,
        1015: 15,
        1016: 16,
        1017: 17,
        1018: 18,
        1019: 19,
        1020: 20,
        1021: 21,
        1022: 22,
        1023: 23,
        1024: 24,
        1025: 25,
        1026: 26,
        1027: 27,
        1028: 28,
        1029: 29,
        1030: 30,
        1031: 31,
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

    def extract(self, command_entry: CommandEntry, feature: np.ndarray) -> None:
        """
        特徴量を抽出する。

        Args:
            command_entry (CommandEntry): 行動選択情報
            feature (np.ndarray): 特徴量埋め込み先
        """
        # 埋め込み
        cursor: int = 0

        # 行動の種類
        self._fill_command_type(feature[cursor : cursor + self.SIZE_COMMAND_TYPE, :, :], command_entry)
        cursor += self.SIZE_COMMAND_TYPE

        # カードID
        self._fill_card_id(feature[cursor : cursor + self.SIZE_CARD_ID, :, :], command_entry)
        cursor += self.SIZE_CARD_ID

        # 対象グリッド
        self._fill_target_grid(feature[cursor : cursor + self.SIZE_TARGET_GRID, :, :], command_entry)
        cursor += self.SIZE_TARGET_GRID

        # 効果番号
        self._fill_effect_no(feature[cursor : cursor + self.SIZE_EFFECT_NO, :, :], command_entry)
        cursor += self.SIZE_EFFECT_NO

        # フェイズ
        self._fill_phase(feature[cursor : cursor + self.SIZE_PHASE, :, :], command_entry)
        cursor += self.SIZE_PHASE

        # 表示形式
        self._fill_stand_turn(feature[cursor : cursor + self.SIZE_STAND_TURN, :, :], command_entry)
        cursor += self.SIZE_STAND_TURN

        # Yes/No
        self._fill_yes_no(feature[cursor : cursor + self.SIZE_YES_NO, :, :], command_entry)
        cursor += self.SIZE_YES_NO

    # =================================================================
    # 埋め込みロジック
    # =================================================================

    def _fill_command_type(self, feature: np.ndarray, command_entry: CommandEntry) -> None:
        """
        行動の種類を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
        """
        command_type: int = command_entry.command_type

        if command_type == CommandType.ACTIVATE:
            feature[0, :, :] = 1.0

        elif command_type == CommandType.ATTACK:
            feature[1, :, :] = 1.0

        elif command_type == CommandType.CHANGE_PHASE:
            feature[2, :, :] = 1.0

        elif command_type == CommandType.DECIDE:
            feature[3, :, :] = 1.0

        elif command_type == CommandType.PASS:
            feature[4, :, :] = 1.0

        elif command_type == CommandType.REVERSE:
            feature[5, :, :] = 1.0

        elif command_type == CommandType.SET:
            feature[6, :, :] = 1.0

        elif command_type == CommandType.SET_MONST:
            feature[7, :, :] = 1.0

        elif command_type == CommandType.SUMMON:
            feature[8, :, :] = 1.0

        elif command_type == CommandType.TURN_ATK:
            feature[9, :, :] = 1.0

        elif command_type == CommandType.TURN_DEF:
            feature[10, :, :] = 1.0

    def _fill_card_id(self, feature: np.ndarray, command_entry: CommandEntry) -> None:
        """
        カードIDを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
        """
        card_id: int = command_entry.card_id

        if card_id == 0:
            feature[0, :, :] = 1.0

        elif card_id in self._CARD_ID_MAP:
            feature[self._CARD_ID_MAP[card_id], :, :] = 1.0

    def _fill_target_grid(self, feature: np.ndarray, command_entry: CommandEntry) -> None:
        """
        対象グリッドを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
        """
        height, width, _ = CardCellLayout.get_coord(
            command_entry.player_id, command_entry.pos_id, command_entry.card_index
        )

        if height != -1:
            feature[0, height, width] = 1.0

    def _fill_effect_no(self, feature: np.ndarray, command_entry: CommandEntry) -> None:
        """
        効果番号を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
        """
        effect_no: int = command_entry.effect_no

        if effect_no == EffectNo.NUM1:
            feature[0, :, :] = 1.0

        elif effect_no == EffectNo.NUM2:
            feature[1, :, :] = 1.0

        elif effect_no == EffectNo.NUM3:
            feature[2, :, :] = 1.0

    def _fill_phase(self, feature: np.ndarray, command_entry: CommandEntry) -> None:
        """
        フェイズを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
        """
        phase: int = command_entry.phase

        if phase == Phase.BATTLE:
            feature[0, :, :] = 1.0

        elif phase == Phase.MAIN2:
            feature[1, :, :] = 1.0

        elif phase == Phase.END:
            feature[2, :, :] = 1.0

    def _fill_stand_turn(self, feature: np.ndarray, command_entry: CommandEntry) -> None:
        """
        表示形式を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
        """
        stand_turn: int = command_entry.stand_turn

        if stand_turn == Turn.VERTICAL:
            feature[0, :, :] = 1.0

        elif stand_turn == Turn.HORIZONTAL:
            feature[1, :, :] = 1.0

    def _fill_yes_no(self, feature: np.ndarray, command_entry: CommandEntry) -> None:
        """
        Yes/Noを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
        """
        yes_no: int = command_entry.yes_no

        if yes_no == YesNo.NO:
            feature[0, :, :] = 1.0

        elif yes_no == YesNo.YES:
            feature[1, :, :] = 1.0
