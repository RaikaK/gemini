from dataclasses import dataclass

from ygo.constants.enums import CommandType, EffectNo, Phase, PlayerId, PosId, Turn, YesNo


@dataclass
class HeadSpec:
    """
    ヘッド仕様
    """

    name: str
    """名称"""

    output_dim: int
    """クラス数 (出力次元数)"""


class ActionHeads:
    """
    アクションヘッド群

    Attributes:
        COMMAND_TYPE (HeadSpec): コマンドタイプヘッド
        CARD_ID (HeadSpec): カードIDヘッド
        PLAYER_ID (HeadSpec): プレイヤーIDヘッド
        POS_ID (HeadSpec): 位置IDヘッド
        EFFECT_NO (HeadSpec): 効果番号ヘッド
        PHASE (HeadSpec): フェイズヘッド
        STAND_TURN (HeadSpec): 表示形式ヘッド
        YES_NO (HeadSpec): Yes/Noヘッド
    """

    # =================================================================
    # マッピング定義
    # =================================================================

    # --- コマンドタイプ (11クラス) ---
    COMMAND_TYPE_MAP: dict[int, int] = {
        CommandType.ACTIVATE: 0,
        CommandType.ATTACK: 1,
        CommandType.CHANGE_PHASE: 2,
        CommandType.DECIDE: 3,
        CommandType.PASS: 4,
        CommandType.REVERSE: 5,
        CommandType.SET: 6,
        CommandType.SET_MONST: 7,
        CommandType.SUMMON: 8,
        CommandType.TURN_ATK: 9,
        CommandType.TURN_DEF: 10,
    }

    # --- カードID (32クラス) ---
    CARD_ID_MAP: dict[int, int] = {
        0: 0,
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

    # --- プレイヤーID (2クラス) ---
    PLAYER_ID_MAP: dict[int, int] = {
        PlayerId.MYSELF: 0,
        PlayerId.RIVAL: 1,
    }

    # --- 位置ID (13クラス) ---
    POS_ID_MAP: dict[int, int] = {
        PosId.MONSTER_L_L: 0,
        PosId.MONSTER_L: 1,
        PosId.MONSTER_C: 2,
        PosId.MONSTER_R: 3,
        PosId.MONSTER_R_R: 4,
        PosId.MAGIC_L_L: 5,
        PosId.MAGIC_L: 6,
        PosId.MAGIC_C: 7,
        PosId.MAGIC_R: 8,
        PosId.MAGIC_R_R: 9,
        PosId.HAND: 10,
        PosId.DECK: 11,
        PosId.GRAVE: 12,
    }

    # --- 効果番号 (3クラス) ---
    EFFECT_NO_MAP: dict[int, int] = {
        EffectNo.NUM1: 0,
        EffectNo.NUM2: 1,
        EffectNo.NUM3: 2,
    }

    # --- フェイズ (3クラス) ---
    PHASE_MAP: dict[int, int] = {
        Phase.BATTLE: 0,
        Phase.MAIN2: 1,
        Phase.END: 2,
    }

    # --- 表示形式 (2クラス) ---
    STAND_TURN_MAP: dict[int, int] = {
        Turn.VERTICAL: 0,
        Turn.HORIZONTAL: 1,
    }

    # --- Yes/No (2クラス) ---
    YES_NO_MAP: dict[int, int] = {
        YesNo.NO: 0,
        YesNo.YES: 1,
    }

    # =================================================================
    # ヘッド定義
    # =================================================================

    COMMAND_TYPE = HeadSpec(name="command_type", output_dim=len(COMMAND_TYPE_MAP))
    CARD_ID = HeadSpec(name="card_id", output_dim=len(CARD_ID_MAP))
    PLAYER_ID = HeadSpec(name="player_id", output_dim=len(PLAYER_ID_MAP))
    POS_ID = HeadSpec(name="pos_id", output_dim=len(POS_ID_MAP))
    EFFECT_NO = HeadSpec(name="effect_no", output_dim=len(EFFECT_NO_MAP))
    PHASE = HeadSpec(name="phase", output_dim=len(PHASE_MAP))
    STAND_TURN = HeadSpec(name="stand_turn", output_dim=len(STAND_TURN_MAP))
    YES_NO = HeadSpec(name="yes_no", output_dim=len(YES_NO_MAP))

    # =================================================================
    # メソッド
    # =================================================================

    @classmethod
    def get_all_heads(cls) -> list[HeadSpec]:
        """
        全てのヘッドを定義順に返す。

        Returns:
            list[HeadSpec]: 全てのヘッド
        """
        return [
            cls.COMMAND_TYPE,
            cls.CARD_ID,
            cls.PLAYER_ID,
            cls.POS_ID,
            cls.EFFECT_NO,
            cls.PHASE,
            cls.STAND_TURN,
            cls.YES_NO,
        ]

    @classmethod
    def to_head_index(cls, head_name: str, value: int) -> int:
        """
        指定されたヘッドの対応するインデックスを返す。

        Args:
            head_name (str): ヘッドの名称
            value (int): 対象の値

        Returns:
            int: 対応するインデックス (対応しない場合は-1)
        """
        if head_name == cls.COMMAND_TYPE.name:
            return cls.COMMAND_TYPE_MAP.get(value, -1)

        elif head_name == cls.CARD_ID.name:
            return cls.CARD_ID_MAP.get(value, -1)

        elif head_name == cls.PLAYER_ID.name:
            return cls.PLAYER_ID_MAP.get(value, -1)

        elif head_name == cls.POS_ID.name:
            return cls.POS_ID_MAP.get(value, -1)

        elif head_name == cls.EFFECT_NO.name:
            return cls.EFFECT_NO_MAP.get(value, -1)

        elif head_name == cls.PHASE.name:
            return cls.PHASE_MAP.get(value, -1)

        elif head_name == cls.STAND_TURN.name:
            return cls.STAND_TURN_MAP.get(value, -1)

        elif head_name == cls.YES_NO.name:
            return cls.YES_NO_MAP.get(value, -1)

        return -1
