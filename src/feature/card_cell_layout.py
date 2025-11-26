from dataclasses import dataclass

from ygo.constants.enums import PlayerId, PosId


@dataclass
class CardCell:
    """
    カードセル
    """

    player_id: int
    """プレイヤーID"""
    pos_id: int
    """位置ID"""
    card_index: int
    """カードインデックス (-1なら指定なし)"""
    is_bag: bool
    """Bagフラグ (集計エリアフラグ)"""


class CardCellLayout:
    """
    カードセルレイアウト

    Attributes:
        MAP (dict[tuple[int, int], CardCell]): 座標からカードセルへのマップ
        REVERSE_MAP (dict[tuple[int, int, int], tuple[int, int, bool]]): カード情報から座標情報へのマップ
    """

    MAP: dict[tuple[int, int], CardCell] = {}
    REVERSE_MAP: dict[tuple[int, int, int], tuple[int, int, bool]] = {}

    @classmethod
    def _register(cls, coord: tuple[int, int], player_id: int, pos_id: int, card_index: int, is_bag: bool) -> None:
        """
        レイアウトを登録する。

        Args:
            coord (tuple[int, int]): 座標 (height, width)
            player_id (int): プレイヤーID
            pos_id (int): 位置ID
            card_index (int): カードインデックス
            is_bag (bool): Bagフラグ
        """
        cls.MAP[coord] = CardCell(player_id, pos_id, card_index, is_bag)
        cls.REVERSE_MAP[(player_id, pos_id, card_index)] = (coord[0], coord[1], is_bag)

    @classmethod
    def get_coord(cls, player_id: int, pos_id: int, card_index: int) -> tuple[int, int, bool]:
        """
        座標情報を取得する。

        Args:
            player_id (int): プレイヤーID
            pos_id (int): 位置ID
            card_index (int): カードインデックス

        Returns:
            tuple[int, int, bool]: 座標情報 (height, width, is_bag)
        """
        # card_index 指定あり
        key: tuple[int, int, int] = (player_id, pos_id, card_index)

        if key in cls.REVERSE_MAP:
            return cls.REVERSE_MAP[key]

        # card_index 指定なし
        wildcard_key: tuple[int, int, int] = (player_id, pos_id, -1)

        if wildcard_key in cls.REVERSE_MAP:
            return cls.REVERSE_MAP[wildcard_key]

        return -1, -1, False


# --- 定義の登録 ---

# ==========================================
# 自分 (MYSELF)
# ==========================================

# H=0: 手札(6~8枚目) & デッキ & 墓地
CardCellLayout._register((0, 0), PlayerId.MYSELF, PosId.HAND, 5, False)
CardCellLayout._register((0, 1), PlayerId.MYSELF, PosId.HAND, 6, False)
CardCellLayout._register((0, 2), PlayerId.MYSELF, PosId.HAND, 7, False)
CardCellLayout._register((0, 3), PlayerId.MYSELF, PosId.DECK, -1, True)
CardCellLayout._register((0, 4), PlayerId.MYSELF, PosId.GRAVE, -1, True)

# H=1: 手札(1~5枚目)
CardCellLayout._register((1, 0), PlayerId.MYSELF, PosId.HAND, 0, False)
CardCellLayout._register((1, 1), PlayerId.MYSELF, PosId.HAND, 1, False)
CardCellLayout._register((1, 2), PlayerId.MYSELF, PosId.HAND, 2, False)
CardCellLayout._register((1, 3), PlayerId.MYSELF, PosId.HAND, 3, False)
CardCellLayout._register((1, 4), PlayerId.MYSELF, PosId.HAND, 4, False)

# H=2: 魔法・罠ゾーン(左~右)
for i in range(5):
    CardCellLayout._register((2, i), PlayerId.MYSELF, PosId.MAGIC_L_L + i, -1, False)

# H=3: メインモンスターゾーン(左~右)
for i in range(5):
    CardCellLayout._register((3, i), PlayerId.MYSELF, PosId.MONSTER_L_L + i, -1, False)

# ==========================================
# 相手 (RIVAL)
# ==========================================

# H=4: メインモンスターゾーン(左~右)
for i in range(5):
    CardCellLayout._register((4, i), PlayerId.RIVAL, PosId.MONSTER_L_L + i, -1, False)

# H=5: 魔法・罠ゾーン(左~右)
for i in range(5):
    CardCellLayout._register((5, i), PlayerId.RIVAL, PosId.MAGIC_L_L + i, -1, False)

# H=6: 手札(1~5枚目)
CardCellLayout._register((6, 0), PlayerId.RIVAL, PosId.HAND, 0, False)
CardCellLayout._register((6, 1), PlayerId.RIVAL, PosId.HAND, 1, False)
CardCellLayout._register((6, 2), PlayerId.RIVAL, PosId.HAND, 2, False)
CardCellLayout._register((6, 3), PlayerId.RIVAL, PosId.HAND, 3, False)
CardCellLayout._register((6, 4), PlayerId.RIVAL, PosId.HAND, 4, False)

# H=7: 手札(6~8枚目) & デッキ & 墓地
CardCellLayout._register((7, 0), PlayerId.RIVAL, PosId.HAND, 5, False)
CardCellLayout._register((7, 1), PlayerId.RIVAL, PosId.HAND, 6, False)
CardCellLayout._register((7, 2), PlayerId.RIVAL, PosId.HAND, 7, False)
CardCellLayout._register((7, 3), PlayerId.RIVAL, PosId.DECK, -1, True)
CardCellLayout._register((7, 4), PlayerId.RIVAL, PosId.GRAVE, -1, True)
