from ygo.constants.enums import ChainState, DmgStepType, EffectNo, Face, PlayerId, Phase, PosId, StepType, Turn
from ygo.models import DuelCard

from src.env.state_data import StateData

from .card_map import CARD_MAP


class Situation:
    """
    状況
    """

    def __init__(self, state: StateData):
        """初期化する。"""
        self._state = state
        self._duel_state_data = state.duel_state_data
        self._duel_card_table = self._duel_state_data.duel_card_table
        self._general_data = self._duel_state_data.general_data
        self._chain_stack = self._duel_state_data.chain_stack
        self._command_request = state.command_request
        self._command_log = self._command_request.command_log

    # ====================================================================================
    # カードカウント (自分)
    # ====================================================================================
    @property
    def my_hand_count(self) -> int:
        """自分の手札の枚数"""
        return sum(
            1 for card in self._duel_card_table if card.player_id == PlayerId.MYSELF and card.pos_id == PosId.HAND
        )

    @property
    def my_deck_count(self) -> int:
        """自分のデッキの枚数"""
        return sum(
            1 for card in self._duel_card_table if card.player_id == PlayerId.MYSELF and card.pos_id == PosId.DECK
        )

    @property
    def my_grave_count(self) -> int:
        """自分の墓地の枚数"""
        return sum(
            1 for card in self._duel_card_table if card.player_id == PlayerId.MYSELF and card.pos_id == PosId.GRAVE
        )

    @property
    def my_grave_monster_count(self) -> int:
        """自分の墓地のモンスター枚数"""
        return sum(
            1
            for card in self._duel_card_table
            if card.player_id == PlayerId.MYSELF
            and card.pos_id == PosId.GRAVE
            and any(data["id"] == card.card_id and data.get("monster") == 1 for data in CARD_MAP.values())
        )

    @property
    def my_mzone_count(self) -> int:
        """自分のモンスターゾーンのカード枚数"""
        return sum(
            1
            for card in self._duel_card_table
            if card.player_id == PlayerId.MYSELF
            and card.pos_id in [PosId.MONSTER_L_L, PosId.MONSTER_L, PosId.MONSTER_C, PosId.MONSTER_R, PosId.MONSTER_R_R]
        )

    @property
    def my_szone_count(self) -> int:
        """自分の魔法＆罠ゾーンのカード枚数"""
        return sum(
            1
            for card in self._duel_card_table
            if card.player_id == PlayerId.MYSELF
            and card.pos_id in [PosId.MAGIC_L_L, PosId.MAGIC_L, PosId.MAGIC_C, PosId.MAGIC_R, PosId.MAGIC_R_R]
        )

    @property
    def my_hand_monster_count(self) -> int:
        """自分の手札のモンスター枚数"""
        return sum(
            1
            for card in self._duel_card_table
            if card.player_id == PlayerId.MYSELF
            and card.pos_id == PosId.HAND
            and any(data["id"] == card.card_id and data.get("monster") == 1 for data in CARD_MAP.values())
        )

    @property
    def my_blue_eyes_hand_count(self) -> int:
        """自分の手札の「青眼の白龍」枚数"""
        return sum(
            1
            for card in self._duel_card_table
            if card.player_id == PlayerId.MYSELF
            and card.pos_id == PosId.HAND
            and card.card_id == CARD_MAP["青眼の白龍"]["id"]
        )

    @property
    def my_blue_eyes_mzone_count(self) -> int:
        """自分のモンスターゾーンの「青眼の白龍」枚数"""
        return sum(
            1
            for card in self._duel_card_table
            if card.player_id == PlayerId.MYSELF
            and card.pos_id
            in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]
            and card.card_id == CARD_MAP["青眼の白龍"]["id"]
        )

    @property
    def my_level4_or_lower_monster_grave_count(self) -> int:
        """自分の墓地のレベル4以下のモンスター枚数"""
        return sum(
            card.player_id == PlayerId.MYSELF
            and card.pos_id == PosId.GRAVE
            and any(
                data["id"] == card.card_id
                and data.get("monster") == 1
                and ((level := data.get("level")) is not None and level <= 4)
                for data in CARD_MAP.values()
            )
            for card in self._duel_card_table
        )

    @property
    def my_level4_or_lower_monster_mzone_count(self) -> int:
        """自分のモンスターゾーンのレベル4以下のモンスター枚数"""
        return sum(
            card.player_id == PlayerId.MYSELF
            and card.pos_id
            in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]
            and any(
                data["id"] == card.card_id
                and data.get("monster") == 1
                and ((level := data.get("level")) is not None and level <= 4)
                for data in CARD_MAP.values()
            )
            for card in self._duel_card_table
        )

    # ====================================================================================
    # カードカウント (相手)
    # ====================================================================================
    @property
    def rival_hand_count(self) -> int:
        """相手の手札の枚数"""
        return sum(
            1 for card in self._duel_card_table if card.player_id == PlayerId.RIVAL and card.pos_id == PosId.HAND
        )

    @property
    def rival_deck_count(self) -> int:
        """相手のデッキの枚数"""
        return sum(
            1 for card in self._duel_card_table if card.player_id == PlayerId.RIVAL and card.pos_id == PosId.DECK
        )

    @property
    def rival_grave_count(self) -> int:
        """相手の墓地の枚数"""
        return sum(
            1 for card in self._duel_card_table if card.player_id == PlayerId.RIVAL and card.pos_id == PosId.GRAVE
        )

    @property
    def rival_mzone_count(self) -> int:
        """相手のモンスターゾーンのカード枚数"""
        return sum(
            1
            for card in self._duel_card_table
            if card.player_id == PlayerId.RIVAL
            and card.pos_id in [PosId.MONSTER_L_L, PosId.MONSTER_L, PosId.MONSTER_C, PosId.MONSTER_R, PosId.MONSTER_R_R]
        )

    @property
    def rival_szone_count(self) -> int:
        """相手の魔法＆罠ゾーンのカード枚数"""
        return sum(
            1
            for card in self._duel_card_table
            if card.player_id == PlayerId.RIVAL
            and card.pos_id in [PosId.MAGIC_L_L, PosId.MAGIC_L, PosId.MAGIC_C, PosId.MAGIC_R, PosId.MAGIC_R_R]
        )

    # ====================================================================================
    # カード盤面 - モンスター (自分)
    # ====================================================================================
    @property
    def my_monster_atk_max(self) -> int:
        """自分のモンスターゾーンの最大攻撃力 (表側表示のみ)"""
        max_val = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.FRONT:
                    if card.atk_val > max_val:
                        max_val = card.atk_val

        return max_val

    @property
    def my_monster_atk_sum(self) -> int:
        """自分のモンスターゾーンの攻撃力合計 (表側表示のみ)"""
        return sum(
            card.atk_val
            for card in self._duel_card_table
            if card.player_id == PlayerId.MYSELF
            and card.pos_id
            in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]
            and card.face == Face.FRONT
        )

    @property
    def my_monster_atk_min_2sum(self) -> int:
        """自分のモンスターゾーンの2体ペアのうち、一番低い合計攻撃力 (表側表示のみ)"""
        atk_list = [
            card.atk_val
            for card in self._duel_card_table
            if card.player_id == PlayerId.MYSELF
            and card.pos_id
            in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]
            and card.face == Face.FRONT
        ]
        if len(atk_list) < 2:
            return 0

        atk_list.sort()

        return atk_list[0] + atk_list[1]

    @property
    def my_monster_def_max(self) -> int:
        """自分のモンスターゾーンの最大防御力 (表側表示のみ)"""
        max_val = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.FRONT:
                    if card.def_val > max_val:
                        max_val = card.def_val

        return max_val

    @property
    def my_hand_max_normal_summon_atk(self) -> int:
        """自分の手札の通常召喚可能なモンスターの最大攻撃力"""
        max_val = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id == PosId.HAND:
                for name, data in CARD_MAP.items():
                    if data["id"] == card.card_id:
                        if data.get("monster") == 1 and data.get("level", 0) <= 4 and name != "白竜の聖騎士":
                            if card.atk_val > max_val:
                                max_val = card.atk_val

                        break

        return max_val

    @property
    def my_grave_monster_atk_max(self) -> int:
        """自分の墓地のモンスターの最大攻撃力"""
        max_val = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id == PosId.GRAVE:
                for data in CARD_MAP.values():
                    if data["id"] == card.card_id and data.get("monster") == 1:
                        if card.atk_val > max_val:
                            max_val = card.atk_val

                        break

        return max_val

    @property
    def my_face_up_atk_monster_count(self) -> int:
        """自分の表側攻撃表示モンスターの数"""
        count = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.FRONT and card.turn == Turn.VERTICAL:
                    count += 1

        return count

    @property
    def my_face_up_def_monster_count(self) -> int:
        """自分の表側守備表示モンスターの数"""
        count = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.FRONT and card.turn == Turn.HORIZONTAL:
                    count += 1

        return count

    @property
    def my_face_down_monster_count(self) -> int:
        """自分の裏側守備表示モンスターの数"""
        count = 0
        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.BACK:
                    count += 1

        return count

    # ====================================================================================
    # カード盤面 - モンスター (相手)
    # ====================================================================================
    @property
    def rival_monster_atk_max(self) -> int:
        """相手のモンスターゾーンの最大攻撃力 (表側表示のみ)"""
        max_val = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.FRONT:
                    if card.atk_val > max_val:
                        max_val = card.atk_val

        return max_val

    @property
    def rival_monster_atk_sum(self) -> int:
        """相手のモンスターゾーンの攻撃力合計 (表側表示のみ)"""
        return sum(
            card.atk_val
            for card in self._duel_card_table
            if card.player_id == PlayerId.RIVAL
            and card.pos_id
            in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]
            and card.face == Face.FRONT
        )

    @property
    def rival_monster_def_max(self) -> int:
        """相手のモンスターゾーンの最大防御力 (表側表示のみ)"""
        max_val = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.FRONT:
                    if card.def_val > max_val:
                        max_val = card.def_val

        return max_val

    @property
    def rival_grave_monster_atk_max(self) -> int:
        """相手の墓地のモンスターの最大攻撃力"""
        max_val = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id == PosId.GRAVE:
                for data in CARD_MAP.values():
                    if data["id"] == card.card_id and data.get("monster") == 1:
                        if card.atk_val > max_val:
                            max_val = card.atk_val

                        break

        return max_val

    @property
    def rival_monster_atk_min(self) -> int:
        """相手のモンスターゾーンの最小攻撃力 (表側表示のみ)"""
        min_val = 99999
        found = False

        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.FRONT:
                    if card.atk_val < min_val:
                        min_val = card.atk_val
                        found = True

        return min_val if found else 0

    @property
    def rival_monster_def_min(self) -> int:
        """相手のモンスターゾーンの最小防御力 (表側表示のみ)"""
        min_val = 99999
        found = False

        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.FRONT:
                    if card.def_val < min_val:
                        min_val = card.def_val
                        found = True

        return min_val if found else 0

    @property
    def rival_face_up_def_monster_def_min(self) -> int:
        """相手の表側守備表示モンスターの最小防御力"""
        min_val = 99999
        found = False

        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.FRONT and card.turn == Turn.HORIZONTAL:
                    if card.def_val < min_val:
                        min_val = card.def_val
                        found = True

        return min_val if found else 0

    @property
    def rival_face_up_atk_monster_count(self) -> int:
        """相手の表側攻撃表示モンスターの数"""
        count = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.FRONT and card.turn == Turn.VERTICAL:
                    count += 1

        return count

    @property
    def rival_face_up_def_monster_count(self) -> int:
        """相手の表側守備表示モンスターの数"""
        count = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.FRONT and card.turn == Turn.HORIZONTAL:
                    count += 1

        return count

    @property
    def rival_face_down_monster_count(self) -> int:
        """相手の裏側守備表示モンスターの数"""
        count = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.face == Face.BACK:
                    count += 1

        return count

    @property
    def rival_blue_eyes_mzone_count(self) -> int:
        """相手のモンスターゾーンの「青眼の白龍」枚数"""
        return sum(
            1
            for card in self._duel_card_table
            if card.player_id == PlayerId.RIVAL
            and card.pos_id
            in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]
            and card.card_id == CARD_MAP["青眼の白龍"]["id"]
        )

    @property
    def rival_blue_eyes_grave_count(self) -> int:
        """相手の墓地の「青眼の白龍」枚数"""
        return sum(
            1
            for card in self._duel_card_table
            if card.player_id == PlayerId.RIVAL
            and card.pos_id == PosId.GRAVE
            and card.card_id == CARD_MAP["青眼の白龍"]["id"]
        )

    def rival_face_up_atk_monster_count_within(self, min_val: int, max_val: int) -> int:
        """
        相手の表側攻撃表示モンスターのうち、攻撃力が[min_val, max_val]の範囲内にある枚数
        """
        count = 0

        for card in self._duel_card_table:
            if (
                card.player_id == PlayerId.RIVAL
                and card.pos_id
                in [
                    PosId.MONSTER_L_L,
                    PosId.MONSTER_L,
                    PosId.MONSTER_C,
                    PosId.MONSTER_R,
                    PosId.MONSTER_R_R,
                ]
                and card.face == Face.FRONT
                and card.turn == Turn.VERTICAL
                and min_val <= card.atk_val <= max_val
            ):
                count += 1

        return count

    def rival_face_up_def_monster_count_within(self, min_val: int, max_val: int) -> int:
        """
        相手の表側守備表示モンスターのうち、守備力が[min_val, max_val]の範囲内にある枚数
        """
        count = 0

        for card in self._duel_card_table:
            if (
                card.player_id == PlayerId.RIVAL
                and card.pos_id
                in [
                    PosId.MONSTER_L_L,
                    PosId.MONSTER_L,
                    PosId.MONSTER_C,
                    PosId.MONSTER_R,
                    PosId.MONSTER_R_R,
                ]
                and card.face == Face.FRONT
                and card.turn == Turn.HORIZONTAL
                and min_val <= card.def_val <= max_val
            ):
                count += 1

        return count

    # ====================================================================================
    # カード盤面 - 魔法＆罠 (自分・相手)
    # ====================================================================================
    @property
    def my_face_down_spell_count(self) -> int:
        """自分のセットされている魔法＆罠カードの数"""
        count = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MAGIC_L_L,
                PosId.MAGIC_L,
                PosId.MAGIC_C,
                PosId.MAGIC_R,
                PosId.MAGIC_R_R,
            ]:
                if card.face == Face.BACK:
                    count += 1

        return count

    @property
    def rival_face_down_spell_count(self) -> int:
        """相手のセットされている魔法＆罠カードの数"""
        count = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MAGIC_L_L,
                PosId.MAGIC_L,
                PosId.MAGIC_C,
                PosId.MAGIC_R,
                PosId.MAGIC_R_R,
            ]:
                if card.face == Face.BACK:
                    count += 1

        return count

    @property
    def my_activatable_spell_count(self) -> int:
        """自分が発動可能な(セットして1ターン経過した)罠・魔法の数"""
        count = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MAGIC_L_L,
                PosId.MAGIC_L,
                PosId.MAGIC_C,
                PosId.MAGIC_R,
                PosId.MAGIC_R_R,
            ]:
                if card.face == Face.BACK and card.turn_passed == 1:
                    count += 1

        return count

    @property
    def rival_activatable_spell_count(self) -> int:
        """相手の発動可能な(セットして1ターン経過した)罠・魔法の数"""
        count = 0

        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MAGIC_L_L,
                PosId.MAGIC_L,
                PosId.MAGIC_C,
                PosId.MAGIC_R,
                PosId.MAGIC_R_R,
            ]:
                if card.face == Face.BACK and card.turn_passed == 1:
                    count += 1

        return count

    # ====================================================================================
    # 戦闘状態
    # ====================================================================================
    @property
    def is_my_monster_attacking(self) -> bool:
        """自分のモンスターが攻撃中か"""
        return self.my_attacking_monster_id != -1

    @property
    def is_my_monster_attacked(self) -> bool:
        """自分のモンスターが攻撃対象に選択されているか"""
        return self.my_attacked_monster_id != -1

    @property
    def is_rival_monster_attacking(self) -> bool:
        """相手のモンスターが攻撃中か"""
        return self.rival_attacking_monster_id != -1

    @property
    def is_rival_monster_attacked(self) -> bool:
        """相手のモンスターが攻撃対象に選択されているか"""
        return self.rival_attacked_monster_id != -1

    @property
    def my_attacking_monster_id(self) -> int:
        """攻撃中の自分のモンスターのcard_idを取得します。存在しない場合は-1。"""
        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.is_attacking == 1:
                    return card.card_id

        return -1

    @property
    def my_attacking_monster_atk(self) -> int:
        """
        攻撃中の自分のモンスターの現在の攻撃力を取得します。存在しない場合は -1 を返します。
        """
        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.is_attacking == 1:
                    return card.atk_val
        return -1

    @property
    def my_attacking_monster_def(self) -> int:
        """
        攻撃中の自分のモンスターの現在の防御力を取得します。存在しない場合は -1 を返します。
        """
        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.is_attacking == 1:
                    return card.def_val
        return -1

    @property
    def my_attacked_monster_id(self) -> int:
        """攻撃対象となっている自分のモンスターのcard_idを取得します。存在しない場合は-1。"""
        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.is_attacked == 1:
                    return card.card_id

        return -1

    @property
    def my_attacked_monster_atk(self) -> int:
        """
        攻撃対象となっている自分のモンスターの現在の攻撃力を取得します。存在しない場合は -1 を返します。
        """
        for card in self._duel_card_table:
            if card.player_id == PlayerId.MYSELF and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.is_attacked == 1:
                    return card.atk_val
        return -1

    @property
    def rival_attacking_monster_id(self) -> int:
        """攻撃中の相手のモンスターのcard_idを取得します。存在しない場合は-1。"""
        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.is_attacking == 1:
                    return card.card_id

        return -1

    @property
    def rival_attacking_monster_atk(self) -> int:
        """
        攻撃中の相手のモンスターの現在の攻撃力を取得します。存在しない場合は -1 を返します。
        """
        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.is_attacking == 1:
                    return card.atk_val
        return -1

    @property
    def rival_attacked_monster_id(self) -> int:
        """攻撃対象となっている相手のモンスターのcard_idを取得します。存在しない場合は-1。"""
        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.is_attacked == 1:
                    return card.card_id

        return -1

    @property
    def rival_attacked_monster_atk(self) -> int:
        """
        攻撃対象となっている相手のモンスターの現在の攻撃力を取得します。存在しない場合は -1 を返します。
        """
        for card in self._duel_card_table:
            if card.player_id == PlayerId.RIVAL and card.pos_id in [
                PosId.MONSTER_L_L,
                PosId.MONSTER_L,
                PosId.MONSTER_C,
                PosId.MONSTER_R,
                PosId.MONSTER_R_R,
            ]:
                if card.is_attacked == 1:
                    return card.atk_val
        return -1

    # ====================================================================================
    # 装備状態
    # ====================================================================================
    def is_my_monster_equipped(self, card_name: str) -> bool:
        """自分の場の特定のモンスターが装備カードの対象になっているか。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        for i, card in enumerate(self._duel_card_table):
            if (
                card.player_id == PlayerId.MYSELF
                and card.card_id == target_id
                and card.pos_id
                in [PosId.MONSTER_L_L, PosId.MONSTER_L, PosId.MONSTER_C, PosId.MONSTER_R, PosId.MONSTER_R_R]
            ):
                if any(other.equip_target == i for other in self._duel_card_table):
                    return True

        return False

    def is_rival_monster_equipped(self, card_name: str) -> bool:
        """相手の場の特定のモンスターが装備カードの対象になっているか。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        for i, card in enumerate(self._duel_card_table):
            if (
                card.player_id == PlayerId.RIVAL
                and card.card_id == target_id
                and card.pos_id
                in [PosId.MONSTER_L_L, PosId.MONSTER_L, PosId.MONSTER_C, PosId.MONSTER_R, PosId.MONSTER_R_R]
            ):
                if any(other.equip_target == i for other in self._duel_card_table):
                    return True

        return False

    # ====================================================================================
    # 強化状態
    # ====================================================================================
    def is_enhanced_my_monster_exists(self, card_name: str) -> bool:
        """自分のモンスターゾーンに、指定したカード名で元々の攻撃力より強化されているモンスターがあるか。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None or target_info["atk"] <= 0:
            return False

        target_id = target_info["id"]
        target_atk = target_info["atk"]

        for card in self._duel_card_table:
            if (
                card.player_id == PlayerId.MYSELF
                and card.pos_id
                in [PosId.MONSTER_L_L, PosId.MONSTER_L, PosId.MONSTER_C, PosId.MONSTER_R, PosId.MONSTER_R_R]
                and card.face == Face.FRONT
                and card.card_id == target_id
            ):
                if card.atk_val > target_atk:
                    return True

        return False

    def is_enhanced_rival_monster_exists(self, card_name: str) -> bool:
        """相手のモンスターゾーンに、指定したカード名で元々の攻撃力より強化されているモンスターがあるか。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None or target_info["atk"] <= 0:
            return False

        target_id = target_info["id"]
        target_atk = target_info["atk"]

        for card in self._duel_card_table:
            if (
                card.player_id == PlayerId.RIVAL
                and card.pos_id
                in [PosId.MONSTER_L_L, PosId.MONSTER_L, PosId.MONSTER_C, PosId.MONSTER_R, PosId.MONSTER_R_R]
                and card.face == Face.FRONT
                and card.card_id == target_id
            ):
                if card.atk_val > target_atk:
                    return True

        return False

    # ====================================================================================
    # 効果使用状態
    # ====================================================================================
    def has_my_card_used_effect1(self, card_name: str) -> bool:
        """自分の特定のカードが①の効果をこのターン既に使用したか。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.player_id == PlayerId.MYSELF and card.card_id == target_id and card.used_effect1 == 1
            for card in self._duel_card_table
        )

    def has_my_card_used_effect2(self, card_name: str) -> bool:
        """自分の特定のカードが②の効果をこのターン既に使用したか。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.player_id == PlayerId.MYSELF and card.card_id == target_id and card.used_effect2 == 1
            for card in self._duel_card_table
        )

    def has_my_card_used_effect3(self, card_name: str) -> bool:
        """自分の特定のカードが③の効果をこのターン既に使用したか。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.player_id == PlayerId.MYSELF and card.card_id == target_id and card.used_effect3 == 1
            for card in self._duel_card_table
        )

    def has_rival_card_used_effect1(self, card_name: str) -> bool:
        """相手の特定のカードが①の効果をこのターン既に使用したか。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.player_id == PlayerId.RIVAL and card.card_id == target_id and card.used_effect1 == 1
            for card in self._duel_card_table
        )

    def has_rival_card_used_effect2(self, card_name: str) -> bool:
        """相手の特定のカードが②の効果をこのターン既に使用したか。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.player_id == PlayerId.RIVAL and card.card_id == target_id and card.used_effect2 == 1
            for card in self._duel_card_table
        )

    def has_rival_card_used_effect3(self, card_name: str) -> bool:
        """相手の特定のカードが③の効果をこのターン既に使用したか。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.player_id == PlayerId.RIVAL and card.card_id == target_id and card.used_effect3 == 1
            for card in self._duel_card_table
        )

    # ====================================================================================
    # カード (検索・特定)
    # ====================================================================================
    def has_card_in_hand(self, card_name: str) -> bool:
        """自分の手札に指定したカードが存在するか"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.card_id == target_id and card.player_id == PlayerId.MYSELF and card.pos_id == PosId.HAND
            for card in self._duel_card_table
        )

    def has_card_in_grave(self, card_name: str) -> bool:
        """自分の墓地に指定したカードが存在するか"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.card_id == target_id and card.player_id == PlayerId.MYSELF and card.pos_id == PosId.GRAVE
            for card in self._duel_card_table
        )

    def has_rival_card_in_grave(self, card_name: str) -> bool:
        """相手の墓地に指定したカードが存在するか"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.card_id == target_id and card.player_id == PlayerId.RIVAL and card.pos_id == PosId.GRAVE
            for card in self._duel_card_table
        )

    def has_card_on_mzone(self, card_name: str) -> bool:
        """自分のモンスターゾーンに指定したカードが存在するか(表裏問わず)"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.card_id == target_id
            and card.player_id == PlayerId.MYSELF
            and card.pos_id in [PosId.MONSTER_L_L, PosId.MONSTER_L, PosId.MONSTER_C, PosId.MONSTER_R, PosId.MONSTER_R_R]
            for card in self._duel_card_table
        )

    def has_rival_card_on_mzone(self, card_name: str) -> bool:
        """相手のモンスターゾーンに指定したカードが存在するか(表裏問わず)"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.card_id == target_id
            and card.player_id == PlayerId.RIVAL
            and card.pos_id in [PosId.MONSTER_L_L, PosId.MONSTER_L, PosId.MONSTER_C, PosId.MONSTER_R, PosId.MONSTER_R_R]
            for card in self._duel_card_table
        )

    def has_face_card_on_mzone(self, card_name: str) -> bool:
        """自分のモンスターゾーンに指定したカードが表側表示で存在するか"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.card_id == target_id
            and card.player_id == PlayerId.MYSELF
            and card.face == Face.FRONT
            and card.pos_id in [PosId.MONSTER_L_L, PosId.MONSTER_L, PosId.MONSTER_C, PosId.MONSTER_R, PosId.MONSTER_R_R]
            for card in self._duel_card_table
        )

    def has_rival_face_card_on_mzone(self, card_name: str) -> bool:
        """相手のモンスターゾーンに指定したカードが表側表示で存在するか"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.card_id == target_id
            and card.player_id == PlayerId.RIVAL
            and card.face == Face.FRONT
            and card.pos_id in [PosId.MONSTER_L_L, PosId.MONSTER_L, PosId.MONSTER_C, PosId.MONSTER_R, PosId.MONSTER_R_R]
            for card in self._duel_card_table
        )

    def has_card_on_szone(self, card_name: str) -> bool:
        """自分の魔法＆罠ゾーンに指定したカードが存在するか(表裏問わず)"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.card_id == target_id
            and card.player_id == PlayerId.MYSELF
            and card.pos_id in [PosId.MAGIC_L_L, PosId.MAGIC_L, PosId.MAGIC_C, PosId.MAGIC_R, PosId.MAGIC_R_R]
            for card in self._duel_card_table
        )

    def has_rival_card_on_szone(self, card_name: str) -> bool:
        """相手の魔法＆罠ゾーンに指定したカードが存在するか(表裏問わず)"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.card_id == target_id
            and card.player_id == PlayerId.RIVAL
            and card.pos_id in [PosId.MAGIC_L_L, PosId.MAGIC_L, PosId.MAGIC_C, PosId.MAGIC_R, PosId.MAGIC_R_R]
            for card in self._duel_card_table
        )

    def has_card_in_deck(self, card_name: str) -> bool:
        """自分のデッキに指定したカードが存在するか"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        return any(
            card.card_id == target_id and card.player_id == PlayerId.MYSELF and card.pos_id == PosId.DECK
            for card in self._duel_card_table
        )

    # ====================================================================================
    # ターン
    # ====================================================================================
    @property
    def is_my_turn(self) -> bool:
        """自分のターンか"""
        return self._general_data.which_turn_now == PlayerId.MYSELF

    @property
    def is_rival_turn(self) -> bool:
        """相手のターンか"""
        return self._general_data.which_turn_now == PlayerId.RIVAL

    @property
    def turn_num(self) -> int:
        """現在のターン数 (1始まり)"""
        return self._general_data.turn_num + 1

    @property
    def is_first_turn(self) -> bool:
        """最初のターン（先攻1ターン目）か"""
        return self.turn_num == 1

    # ====================================================================================
    # LP
    # ====================================================================================
    @property
    def my_lp(self) -> int:
        """自分のLP"""
        return self._general_data.lp[PlayerId.MYSELF]

    @property
    def rival_lp(self) -> int:
        """相手のLP"""
        return self._general_data.lp[PlayerId.RIVAL]

    @property
    def lp_diff(self) -> int:
        """LPの差分 (自分 - 相手)"""
        return self.my_lp - self.rival_lp

    @property
    def lp_ratio(self) -> float:
        """LPの比率 (自分 / 相手)"""
        return self.my_lp / max(1, self.rival_lp)

    @property
    def is_lp_advantage(self) -> bool:
        """LPで勝っているか"""
        return self.lp_diff > 0

    @property
    def is_lp_disadvantage(self) -> bool:
        """LPで負けているか"""
        return self.lp_diff < 0

    @property
    def is_lp_equal(self) -> bool:
        """LPが同点か"""
        return self.lp_diff == 0

    # ====================================================================================
    # フェイズ
    # ====================================================================================
    @property
    def phase(self) -> int:
        """現在のフェイズ"""
        return self._general_data.current_phase

    @property
    def is_draw_phase(self) -> bool:
        """ドローフェイズか"""
        return self.phase == Phase.DRAW

    @property
    def is_standby_phase(self) -> bool:
        """スタンバイフェイズか"""
        return self.phase == Phase.STANDBY

    @property
    def is_main1(self) -> bool:
        """メインフェイズ1か"""
        return self.phase == Phase.MAIN1

    @property
    def is_battle_phase(self) -> bool:
        """バトルフェイズか"""
        return self.phase == Phase.BATTLE

    @property
    def is_main2(self) -> bool:
        """メインフェイズ2か"""
        return self.phase == Phase.MAIN2

    @property
    def is_end_phase(self) -> bool:
        """エンドフェイズか"""
        return self.phase == Phase.END

    @property
    def is_phase_none(self) -> bool:
        """フェイズ無しか"""
        return self.phase == Phase.NULL

    @property
    def is_main_phase(self) -> bool:
        """メインフェイズ（1または2）か"""
        return self.is_main1 or self.is_main2

    # ====================================================================================
    # ステップ
    # ====================================================================================
    @property
    def step(self) -> int:
        """現在のステップ"""
        return self._general_data.current_step

    @property
    def is_step_null(self) -> bool:
        """ステップ無し (バトルフェイズ以外) か"""
        return self.step == StepType.NULL

    @property
    def is_step_start(self) -> bool:
        """スタートステップか"""
        return self.step == StepType.START

    @property
    def is_step_battle(self) -> bool:
        """バトルステップか"""
        return self.step == StepType.BATTLE

    @property
    def is_step_damage(self) -> bool:
        """ダメージステップか"""
        return self.step == StepType.DAMAGE

    @property
    def is_step_end(self) -> bool:
        """エンドステップか"""
        return self.step == StepType.END

    # ====================================================================================
    # ダメージステップ
    # ====================================================================================
    @property
    def dmg_step(self) -> int:
        """現在のダメージステップ"""
        return self._general_data.current_damage_step

    @property
    def is_dmg_null(self) -> bool:
        """ダメージステップ無し (ダメージステップ以外) か"""
        return self.dmg_step == DmgStepType.NULL

    @property
    def is_dmg_start(self) -> bool:
        """ダメージステップ開始時か"""
        return self.dmg_step == DmgStepType.START

    @property
    def is_dmg_before_calc(self) -> bool:
        """ダメージ計算前か"""
        return self.dmg_step == DmgStepType.BEFORE_CALC

    @property
    def is_dmg_calc(self) -> bool:
        """ダメージ計算時か"""
        return self.dmg_step == DmgStepType.DAMAGE_CALC

    @property
    def is_dmg_after_calc(self) -> bool:
        """ダメージ計算後か"""
        return self.dmg_step == DmgStepType.AFTER_CALC

    @property
    def is_dmg_end(self) -> bool:
        """ダメージステップ終了時か"""
        return self.dmg_step == DmgStepType.END

    # ====================================================================================
    # 召喚権
    # ====================================================================================
    @property
    def my_summon_count(self) -> int:
        """自分の召喚可能回数"""
        return self._general_data.summon_num[PlayerId.MYSELF]

    @property
    def rival_summon_count(self) -> int:
        """相手の召喚可能回数"""
        return self._general_data.summon_num[PlayerId.RIVAL]

    @property
    def can_summon(self) -> bool:
        """自分が召喚可能か"""
        return self.my_summon_count > 0

    @property
    def can_rival_summon(self) -> bool:
        """相手が召喚可能か"""
        return self.rival_summon_count > 0

    # ====================================================================================
    # チェーン (基本情報)
    # ====================================================================================
    @property
    def chain_count(self) -> int:
        """現在のチェーン数"""
        return len(self._chain_stack)

    @property
    def is_chaining(self) -> bool:
        """チェーンが発生しているか"""
        return self.chain_count > 0

    @property
    def last_chain(self):
        """最後のチェーンデータ。スタックが空ならNone。"""
        return self._chain_stack[-1] if self.is_chaining else None

    # ====================================================================================
    # チェーン (最後のチェーン)
    # ====================================================================================
    @property
    def last_chain_card_id(self) -> int:
        """最後に発動したカードのID。スタックが空なら-1。"""
        return self.last_chain.card_id if self.last_chain else -1

    @property
    def last_chain_player_id(self) -> int:
        """最後に効果を発動したプレイヤーのID。スタックが空なら-1。"""
        if not self.last_chain:
            return -1

        target_table_idx = self.last_chain.table_index

        if 0 <= target_table_idx < len(self._duel_state_data.duel_card_table):
            return self._duel_state_data.duel_card_table[target_table_idx].player_id

        return -1

    @property
    def is_last_chain_my(self) -> bool:
        """最後に発動したのが自分の効果か"""
        return self.last_chain_player_id == PlayerId.MYSELF

    @property
    def is_last_chain_rival(self) -> bool:
        """最後に発動したのが相手の効果か"""
        return self.last_chain_player_id == PlayerId.RIVAL

    @property
    def is_last_chain_effect_1(self) -> bool:
        """最後に発動したのが①の効果か。スタックが空ならFalse。"""
        return self.last_chain.effect_no == EffectNo.NUM1 if self.last_chain else False

    @property
    def is_last_chain_effect_2(self) -> bool:
        """最後に発動したのが②の効果か。スタックが空ならFalse。"""
        return self.last_chain.effect_no == EffectNo.NUM2 if self.last_chain else False

    @property
    def is_last_chain_effect_3(self) -> bool:
        """最後に発動したのが③の効果か。スタックが空ならFalse。"""
        return self.last_chain.effect_no == EffectNo.NUM3 if self.last_chain else False

    @property
    def is_last_chain_before_activation(self) -> bool:
        """最後のチェーンが積まれる前の処理中か。スタックが空ならFalse。"""
        return self.last_chain.chain_state == ChainState.SET if self.last_chain else False

    @property
    def is_last_chain_before_resolution(self) -> bool:
        """最後のチェーンが積まれてから効果処理までの間か。スタックが空ならFalse。"""
        return self.last_chain.chain_state == ChainState.WAIT if self.last_chain else False

    @property
    def is_last_chain_resolving(self) -> bool:
        """最後のチェーンが効果処理中か。スタックが空ならFalse。"""
        return self.last_chain.chain_state == ChainState.RESOLVE if self.last_chain else False

    # ====================================================================================
    # チェーン (最初のチェーン)
    # ====================================================================================
    @property
    def first_chain_card_id(self) -> int:
        """最初のチェーンの発動カードのID。スタックが空なら-1。"""
        return self._chain_stack[0].card_id if self.is_chaining else -1

    @property
    def first_chain_player_id(self) -> int:
        """最初のチェーンを発動したプレイヤーのID。スタックが空なら-1。"""
        if not self.is_chaining:
            return -1

        target_table_idx = self._chain_stack[0].table_index

        if 0 <= target_table_idx < len(self._duel_state_data.duel_card_table):
            return self._duel_state_data.duel_card_table[target_table_idx].player_id

        return -1

    @property
    def is_first_chain_my(self) -> bool:
        """最初のチェーンが自分の効果か"""
        return self.first_chain_player_id == PlayerId.MYSELF

    @property
    def is_first_chain_rival(self) -> bool:
        """最初のチェーンが相手の効果か"""
        return self.first_chain_player_id == PlayerId.RIVAL

    # ====================================================================================
    # チェーン (ターゲット)
    # ====================================================================================
    @property
    def last_chain_targets(self) -> list[int]:
        """最後に発動した効果の対象となっているtableIndexのリスト。スタックが空なら空リスト。"""
        return self.last_chain.target_table_index_list if self.last_chain else []

    @property
    def is_my_card_targeted(self) -> bool:
        """最後に発動したチェーンで自分のカードが対象に取られているか。スタックが空ならFalse。"""
        if not self.last_chain:
            return False

        for target_idx in self.last_chain_targets:
            if 0 <= target_idx < len(self._duel_state_data.duel_card_table):
                if self._duel_state_data.duel_card_table[target_idx].player_id == PlayerId.MYSELF:
                    return True

        return False

    @property
    def is_rival_card_targeted(self) -> bool:
        """最後に発動したチェーンで相手のカードが対象に取られているか。スタックが空ならFalse。"""
        if not self.last_chain:
            return False

        for target_idx in self.last_chain_targets:
            if 0 <= target_idx < len(self._duel_state_data.duel_card_table):
                if self._duel_state_data.duel_card_table[target_idx].player_id == PlayerId.RIVAL:
                    return True

        return False

    @property
    def is_my_card_targeted_anywhere(self) -> bool:
        """スタック内のいずれかのチェーンで自分のカードが対象に取られているか。スタックが空ならFalse。"""
        if not self.is_chaining:
            return False

        for chain in self._chain_stack:
            for target_idx in chain.target_table_index_list:
                if 0 <= target_idx < len(self._duel_state_data.duel_card_table):
                    if self._duel_state_data.duel_card_table[target_idx].player_id == PlayerId.MYSELF:
                        return True

        return False

    @property
    def is_rival_card_targeted_anywhere(self) -> bool:
        """スタック内のいずれかのチェーンで相手のカードが対象に取られているか。スタックが空ならFalse。"""
        if not self.is_chaining:
            return False

        for chain in self._chain_stack:
            for target_idx in chain.target_table_index_list:
                if 0 <= target_idx < len(self._duel_state_data.duel_card_table):
                    if self._duel_state_data.duel_card_table[target_idx].player_id == PlayerId.RIVAL:
                        return True

        return False

    # ====================================================================================
    # チェーン (検索・特定)
    # ====================================================================================
    def has_card_in_chain_stack(self, card_name: str) -> bool:
        """特定のカードがチェーンスタックの中に含まれているか。スタックが空ならFalse。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None or not self.is_chaining:
            return False

        target_id = target_info["id"]

        return any(chain.card_id == target_id for chain in self._chain_stack)

    def has_rival_card_in_chain_stack(self, card_name: str) -> bool:
        """特定のカードが相手のチェーンスタックの中に含まれているか。スタックが空ならFalse。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None or not self.is_chaining:
            return False

        target_id = target_info["id"]

        for chain in self._chain_stack:
            if chain.card_id == target_id:
                target_table_idx = chain.table_index

                if 0 <= target_table_idx < len(self._duel_state_data.duel_card_table):
                    if self._duel_state_data.duel_card_table[target_table_idx].player_id == PlayerId.RIVAL:
                        return True

        return False

    def get_chain_link_num(self, card_name: str) -> int:
        """特定のカードがチェーンスタックの何番目に存在するか。スタックが空なら-1。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None or not self.is_chaining:
            return -1

        target_id = target_info["id"]

        for chain in reversed(self._chain_stack):
            if chain.card_id == target_id:
                return chain.chain_num

        return -1

    def has_my_card_targeted_anywhere(self, card_name: str) -> bool:
        """スタック内のいずれかのチェーンで自分の特定のカードが対象に取られているか。スタックが空ならFalse。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None or not self.is_chaining:
            return False

        target_id = target_info["id"]

        for chain in self._chain_stack:
            for target_idx in chain.target_table_index_list:
                if 0 <= target_idx < len(self._duel_state_data.duel_card_table):
                    card = self._duel_state_data.duel_card_table[target_idx]
                    if card.player_id == PlayerId.MYSELF and card.card_id == target_id:
                        return True

        return False

    # ====================================================================================
    # 行動ログ
    # ====================================================================================
    @property
    def log_count(self) -> int:
        """記録されている行動ログの総数"""
        return len(self._command_log)

    def get_log_entry(self, back_index: int = 1):
        """
        最新から指定した数だけ遡った行動ログ(CommandLogEntry)を取得する。back_index=1 が最新。存在しない場合は None。
        """
        if 1 <= back_index <= self.log_count:
            return self._command_log[-back_index]
        return None

    def get_log_command(self, back_index: int = 1):
        """最新から遡った CommandEntry を取得する。back_index=1 が最新。存在しない場合は None。"""
        entry = self.get_log_entry(back_index)

        return entry.command if entry else None

    @property
    def last_action_card_id(self) -> int:
        """直前に行動したカードのID。ログがなければ-1。"""
        command = self.get_log_command(1)

        return command.card_id if command else -1

    @property
    def last_action_table_index(self) -> int:
        """直前に行動したカードのtableインデックス。ログがなければ-1。"""
        command = self.get_log_command(1)
        return command.table_index if command else -1

    @property
    def last_action_selection_type(self) -> int:
        """直前の行動時の SelectionType。ログがなければ-1。"""
        entry = self.get_log_entry(1)
        return entry.selection_type if entry else -1

    @property
    def last_action_selection_id(self) -> int:
        """直前の行動時の SelectionID。ログがなければ-1。"""
        entry = self.get_log_entry(1)
        return entry.selection_id if entry else -1

    def was_action_taken_by_card(self, card_name: str, back_count: int = 3) -> bool:
        """直近 n 件のログの中で、特定のカードが何らかのアクションを起こしたか。"""
        target_info = CARD_MAP.get(card_name)

        if target_info is None:
            return False

        target_id = target_info["id"]

        check_range = min(back_count, self.log_count)

        for i in range(1, check_range + 1):
            command = self.get_log_command(i)

            if command and command.card_id == target_id:
                return True

        return False

    # ====================================================================================
    # 直前行動カード
    # ====================================================================================
    def _get_last_action_card(self) -> DuelCard | None:
        """直前に行動したカードのを取得する。"""
        index = self.last_action_table_index

        if 0 <= index < len(self._duel_card_table):

            return self._duel_card_table[index]

        return None

    @property
    def last_action_card_face(self) -> int:
        """直前に行動したカードの表裏 (0:表, 1:裏)。なければ-1。"""
        card = self._get_last_action_card()
        return card.face if card else -1

    @property
    def last_action_card_turn(self) -> int:
        """直前に行動したカードの攻守向き (0:攻撃, 1:守備)。なければ-1。"""
        card = self._get_last_action_card()
        return card.turn if card else -1

    @property
    def last_action_card_atk(self) -> int:
        """直前に行動したカードの現在の攻撃力。なければ-1。"""
        card = self._get_last_action_card()
        return card.atk_val if card else -1

    @property
    def last_action_card_def(self) -> int:
        """直前に行動したカードの現在の防御力。なければ-1。"""
        card = self._get_last_action_card()
        return card.def_val if card else -1

    @property
    def last_action_card_used_effect1(self) -> bool:
        """直前に行動したカードが①の効果を使用済みか。なければFalse。"""
        card = self._get_last_action_card()
        return (card.used_effect1 == 1) if card else False

    @property
    def last_action_card_used_effect2(self) -> bool:
        """直前に行動したカードが②の効果を使用済みか。なければFalse。"""
        card = self._get_last_action_card()
        return (card.used_effect2 == 1) if card else False

    @property
    def last_action_card_equip_target(self) -> int:
        """直前に行動したカードの装備対象のtableIndex。なければ-1。"""
        card = self._get_last_action_card()
        return card.equip_target if card else -1
