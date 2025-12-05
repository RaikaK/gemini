import numpy as np

from ygo.constants.enums import CommandType, EffectNo, Phase, Turn, YesNo
from ygo.models.command_request import CommandEntry

from src.feature.card_cell_layout import CardCellLayout
from src.feature.card_data import CARD_MAP


class EntryExtractor:
    """
    行動選択特徴量抽出器
    """

    # --- チャンネルサイズ ---
    SIZE_COMMAND_TYPE = 11
    SIZE_CARD_ID = 32
    SIZE_EFFECT_NO = 3
    SIZE_PHASE = 3
    SIZE_STAND_TURN = 2
    SIZE_YES_NO = 2

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
        # 座標の取得
        height, width, _ = CardCellLayout.get_coord(
            command_entry.player_id, command_entry.pos_id, command_entry.card_index
        )

        # 埋め込み
        cursor: int = 0

        # コマンドタイプ
        self._fill_command_type(feature[cursor : cursor + self.SIZE_COMMAND_TYPE, :, :], command_entry, height, width)
        cursor += self.SIZE_COMMAND_TYPE

        # カードID
        self._fill_card_id(feature[cursor : cursor + self.SIZE_CARD_ID, :, :], command_entry, height, width)
        cursor += self.SIZE_CARD_ID

        # 効果番号
        self._fill_effect_no(feature[cursor : cursor + self.SIZE_EFFECT_NO, :, :], command_entry, height, width)
        cursor += self.SIZE_EFFECT_NO

        # フェイズ
        self._fill_phase(feature[cursor : cursor + self.SIZE_PHASE, :, :], command_entry, height, width)
        cursor += self.SIZE_PHASE

        # 表示形式
        self._fill_stand_turn(feature[cursor : cursor + self.SIZE_STAND_TURN, :, :], command_entry, height, width)
        cursor += self.SIZE_STAND_TURN

        # Yes/No
        self._fill_yes_no(feature[cursor : cursor + self.SIZE_YES_NO, :, :], command_entry, height, width)
        cursor += self.SIZE_YES_NO

    # =================================================================
    # 埋め込みロジック
    # =================================================================

    def _set_value(self, feature: np.ndarray, channel: int, height: int, width: int) -> None:
        """
        座標指定に応じて値をセットする。
        """
        if height != -1:
            feature[channel, height, width] = 1.0

        else:
            feature[channel, :, :] = 1.0

    def _fill_command_type(self, feature: np.ndarray, command_entry: CommandEntry, height: int, width: int) -> None:
        """
        コマンドタイプを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
            height (int): height
            width (int): width
        """
        command_type: int = command_entry.command_type

        if command_type == CommandType.ACTIVATE:
            self._set_value(feature, 0, height, width)

        elif command_type == CommandType.ATTACK:
            self._set_value(feature, 1, height, width)

        elif command_type == CommandType.CHANGE_PHASE:
            self._set_value(feature, 2, height, width)

        elif command_type == CommandType.DECIDE:
            self._set_value(feature, 3, height, width)

        elif command_type == CommandType.PASS:
            self._set_value(feature, 4, height, width)

        elif command_type == CommandType.REVERSE:
            self._set_value(feature, 5, height, width)

        elif command_type == CommandType.SET:
            self._set_value(feature, 6, height, width)

        elif command_type == CommandType.SET_MONST:
            self._set_value(feature, 7, height, width)

        elif command_type == CommandType.SUMMON:
            self._set_value(feature, 8, height, width)

        elif command_type == CommandType.TURN_ATK:
            self._set_value(feature, 9, height, width)

        elif command_type == CommandType.TURN_DEF:
            self._set_value(feature, 10, height, width)

    def _fill_card_id(self, feature: np.ndarray, command_entry: CommandEntry, height: int, width: int) -> None:
        """
        カードIDを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
            height (int): height
            width (int): width
        """
        card_id: int = command_entry.card_id

        if card_id == 0:
            self._set_value(feature, 0, height, width)

        elif card_id in CARD_MAP:
            self._set_value(feature, CARD_MAP[card_id]["idx"], height, width)

    # _fill_target_grid 削除

    def _fill_effect_no(self, feature: np.ndarray, command_entry: CommandEntry, height: int, width: int) -> None:
        """
        効果番号を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
            height (int): height
            width (int): width
        """
        effect_no: int = command_entry.effect_no

        if effect_no == EffectNo.NUM1:
            self._set_value(feature, 0, height, width)

        elif effect_no == EffectNo.NUM2:
            self._set_value(feature, 1, height, width)

        elif effect_no == EffectNo.NUM3:
            self._set_value(feature, 2, height, width)

    def _fill_phase(self, feature: np.ndarray, command_entry: CommandEntry, height: int, width: int) -> None:
        """
        フェイズを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
            height (int): height
            width (int): width
        """
        phase: int = command_entry.phase

        if phase == Phase.BATTLE:
            self._set_value(feature, 0, height, width)

        elif phase == Phase.MAIN2:
            self._set_value(feature, 1, height, width)

        elif phase == Phase.END:
            self._set_value(feature, 2, height, width)

    def _fill_stand_turn(self, feature: np.ndarray, command_entry: CommandEntry, height: int, width: int) -> None:
        """
        表示形式を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
            height (int): height
            width (int): width
        """
        stand_turn: int = command_entry.stand_turn

        if stand_turn == Turn.VERTICAL:
            self._set_value(feature, 0, height, width)

        elif stand_turn == Turn.HORIZONTAL:
            self._set_value(feature, 1, height, width)

    def _fill_yes_no(self, feature: np.ndarray, command_entry: CommandEntry, height: int, width: int) -> None:
        """
        Yes/Noを埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_entry (CommandEntry): 行動選択情報
            height (int): height
            width (int): width
        """
        yes_no: int = command_entry.yes_no

        if yes_no == YesNo.NO:
            self._set_value(feature, 0, height, width)

        elif yes_no == YesNo.YES:
            self._set_value(feature, 1, height, width)
