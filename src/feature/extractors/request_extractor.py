import numpy as np

from ygo.constants.enums import SelectionType
from ygo.constants.selection_id import SelectionId
from ygo.models.command_request import CommandRequest


class RequestExtractor:
    """
    行動要求特徴量抽出器
    """

    # --- チャンネルサイズ ---
    SIZE_SELECTION_TYPE = 4
    SIZE_SELECTION_ID = 18

    def __init__(self, scaling_factor: float) -> None:
        """
        初期化する。

        Args:
            scaling_factor (float): スケーリング係数

        Attributes:
            scaling_factor (float): スケーリング係数
        """
        self.scaling_factor: float = scaling_factor

    def extract(self, command_request: CommandRequest, feature: np.ndarray) -> None:
        """
        特徴量を抽出する。

        Args:
            command_request (CommandRequest): 行動要求情報
            feature (np.ndarray): 特徴量埋め込み先
        """
        # 埋め込み
        cursor: int = 0

        # 行動要求の種類
        self._fill_selection_type(feature[cursor : cursor + self.SIZE_SELECTION_TYPE, :, :], command_request)
        cursor += self.SIZE_SELECTION_TYPE

        # 行動要求の説明
        self._fill_selection_id(feature[cursor : cursor + self.SIZE_SELECTION_ID, :, :], command_request)
        cursor += self.SIZE_SELECTION_ID

    # =================================================================
    # 埋め込みロジック
    # =================================================================

    def _fill_selection_type(self, feature: np.ndarray, command_requet: CommandRequest) -> None:
        """
        行動要求の種類を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_request (CommandRequest): 行動要求情報
        """
        selection_type: int = command_requet.selection_type

        if selection_type == SelectionType.SELECT_ATTACK_TARGET:
            feature[0, :, :] = 1.0

        elif selection_type == SelectionType.CHECK_ACTIVATION:
            feature[1, :, :] = 1.0

        elif selection_type == SelectionType.SUMMONING:
            feature[2, :, :] = 1.0

        elif selection_type == SelectionType.OTHER:
            feature[3, :, :] = 1.0

    def _fill_selection_id(self, feature: np.ndarray, command_request: CommandRequest) -> None:
        """
        行動要求の説明を埋め込む。

        Args:
            feature (np.ndarray): 特徴量埋め込み先
            command_request (CommandRequest): 行動要求情報
        """
        selection_id: int = command_request.selection_id

        if selection_id == SelectionId.SELECT_CARD_AS_TRIBUTE:
            feature[0, :, :] = 1.0

        elif selection_id == SelectionId.SET_A_SPELL_OR_TRAP_CARD_ON_THE_FIELD_Q:
            feature[1, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_SPELL_OR_TRAP_CARD_TO_SET_ON_FIELD:
            feature[2, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_MONSTER_TO_SPECIAL_SUMMON_FROM_YOUR_DECK:
            feature[3, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_CARD_TO_ADD_FROM_YOUR_DECK_TO_YOUR_HAND:
            feature[4, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_CARD_TO_SEND_TO_GRAVEYARD:
            feature[5, :, :] = 1.0

        elif selection_id == SelectionId.SEND_A_CARD_IN_YOUR_HAND_TO_THE_GRAVEYARD:
            feature[6, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_CARD_TO_TARGET:
            feature[7, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_MONSTER:
            feature[8, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_CARD_TO_DESTROY:
            feature[9, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_NECESSARY_MONSTER_TO_MATCH_REQUIRED_NUMBER_OF_LEVEL:
            feature[10, :, :] = 1.0

        elif selection_id == SelectionId.TRIBUTE_NECESSARY_MONSTER_TO_MATCH_REQUIRED_NUMBER_OF_LEVEL:
            feature[11, :, :] = 1.0

        elif selection_id == SelectionId.CONTINUE_TO_ATTACK_Q:
            feature[12, :, :] = 1.0

        elif selection_id == SelectionId.DISCARD_FROM_YOUR_HAND:
            feature[13, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_BATTLE_POSITION_OF_CARD:
            feature[14, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_MONSTER_TO_SWITCH_TO_FACEDOWN_DEFENSE_POSITION:
            feature[15, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_MONSTER_TO_SPECIAL_SUMMON:
            feature[16, :, :] = 1.0

        elif selection_id == SelectionId.SELECT_MONSTER_FROM_HAND_TO_SPECIAL_SUMMON:
            feature[17, :, :] = 1.0
