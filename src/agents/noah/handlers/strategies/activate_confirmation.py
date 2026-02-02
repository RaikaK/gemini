import random
from typing import Callable

from ygo.constants.enums import PosId
from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.activate_confirmation import OPTIONS
from ..utils import CARD_MAP, Situation, write_debug_log


def select_activate_confirmation(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """発動確認 (SelectionType:5) & 値無し (SelectionId:-1)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        (
            "card_id:サイクロン|command_type:ACTIVATE",
            lambda s: s.has_rival_card_in_chain_stack("リビングデッドの呼び声")
            or s.has_rival_card_in_chain_stack("強化蘇生")
            or s.has_rival_card_in_chain_stack("早すぎた埋葬"),
        ),
        (
            "card_id:サイクロン|command_type:ACTIVATE",
            lambda s: s.is_rival_turn and (s.is_end_phase or s.is_phase_none) and s.rival_szone_count >= 1,
        ),
        (
            "card_id:サイクロン|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.rival_szone_count >= 1
            and (
                (
                    s.has_card_in_hand("白竜降臨")
                    and s.has_card_in_hand("白竜の聖騎士")
                    and (s.my_mzone_count >= 1 or s.my_hand_monster_count >= 2)
                )
                or (
                    s.has_card_in_hand("高等儀式術")
                    and s.has_card_in_hand("白竜の聖騎士")
                    and (
                        s.has_card_in_deck("アレキサンドライドラゴン")
                        or s.has_card_in_deck("洞窟に潜む竜")
                        or s.has_card_in_deck("サファイアドラゴン")
                    )
                )
            ),
        ),
        (
            "card_id:砂塵の大竜巻|command_type:ACTIVATE",
            lambda s: s.has_rival_card_in_chain_stack("リビングデッドの呼び声")
            or s.has_rival_card_in_chain_stack("強化蘇生")
            or s.has_rival_card_in_chain_stack("早すぎた埋葬"),
        ),
        (
            "card_id:砂塵の大竜巻|command_type:ACTIVATE",
            lambda s: s.is_rival_turn and (s.is_end_phase or s.is_phase_none) and s.rival_szone_count >= 1,
        ),
        (
            "card_id:砂塵の大竜巻|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.rival_szone_count >= 1
            and (
                (
                    s.has_card_in_hand("白竜降臨")
                    and s.has_card_in_hand("白竜の聖騎士")
                    and (s.my_mzone_count >= 1 or s.my_hand_monster_count >= 2)
                )
                or (
                    s.has_card_in_hand("高等儀式術")
                    and s.has_card_in_hand("白竜の聖騎士")
                    and (
                        s.has_card_in_deck("アレキサンドライドラゴン")
                        or s.has_card_in_deck("洞窟に潜む竜")
                        or s.has_card_in_deck("サファイアドラゴン")
                    )
                )
            ),
        ),
        (
            "card_id:禁じられた聖槍|command_type:ACTIVATE",
            lambda s: s.is_chaining
            and s.has_face_card_on_mzone("青眼の白龍")
            and (
                s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－")
                or s.has_card_in_chain_stack("激流葬")
                or s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            ),
        ),
        (
            "card_id:禁じられた聖槍|command_type:ACTIVATE",
            lambda s: s.is_chaining
            and (s.my_lp <= 2000 or s.rival_monster_atk_max >= s.my_lp)
            and s.my_mzone_count >= 1
            and s.is_last_chain_rival,
        ),
        (
            "card_id:禁じられた聖槍|command_type:ACTIVATE",
            lambda s: s.is_chaining
            and s.is_my_monster_attacking
            and (
                s.my_attacking_monster_id == CARD_MAP["青眼の白龍"]["id"]
                or s.my_attacking_monster_id == CARD_MAP["白竜の聖騎士"]["id"]
                or s.my_attacking_monster_id == CARD_MAP["アサルトワイバーン"]["id"]
            )
            and (s.my_attacking_monster_atk - 800) >= s.rival_attacked_monster_atk
            and s.is_last_chain_rival,
        ),
        (
            "card_id:禁じられた聖槍|command_type:ACTIVATE",
            lambda s: s.is_chaining
            and s.is_last_chain_rival
            and s.is_my_card_targeted
            and s.has_face_card_on_mzone("青眼の白龍"),
        ),
        (
            "card_id:禁じられた聖槍|command_type:ACTIVATE",
            lambda s: s.is_battle_phase
            and s.is_dmg_before_calc
            and (
                (
                    s.is_my_turn
                    and s.my_attacking_monster_atk <= s.rival_attacked_monster_atk
                    and s.my_attacking_monster_atk > (s.rival_attacked_monster_atk - 800)
                )
                or (
                    s.is_rival_turn
                    and s.my_attacked_monster_atk <= s.rival_attacking_monster_atk
                    and s.my_attacked_monster_atk > (s.rival_attacking_monster_atk - 800)
                )
            ),
        ),
        (
            "card_id:激流葬|command_type:ACTIVATE",
            lambda s: s.rival_mzone_count >= 1 and (s.rival_monster_atk_max >= s.my_lp or s.my_lp <= 1500),
        ),
        (
            "card_id:激流葬|command_type:ACTIVATE",
            lambda s: s.rival_mzone_count >= 3
            and (
                s.rival_mzone_count > s.my_mzone_count
                or s.rival_monster_atk_max > s.my_monster_atk_max
                or (s.is_my_turn and s.is_main1 and s.can_summon and s.my_hand_count > s.rival_hand_count)
            ),
        ),
        (
            "card_id:激流葬|command_type:ACTIVATE",
            lambda s: s.rival_mzone_count >= 2
            and (
                not s.has_face_card_on_mzone("青眼の白龍")
                or (
                    s.rival_monster_atk_max >= 2500
                    and (
                        (
                            s.is_my_turn
                            and (
                                s.has_card_in_hand("銀龍の轟咆")
                                or s.has_card_on_szone("銀龍の轟咆")
                                or s.has_card_in_hand("早すぎた埋葬")
                                or s.has_card_on_szone("リビングデッドの呼び声")
                            )
                        )
                        or (
                            not s.is_my_turn
                            and (s.has_card_on_szone("銀龍の轟咆") or s.has_card_on_szone("リビングデッドの呼び声"))
                        )
                    )
                )
            ),
        ),
        (
            "card_id:激流葬|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.is_main1
            and not s.has_face_card_on_mzone("青眼の白龍")
            and (s.rival_mzone_count >= 2 or (s.rival_mzone_count >= 1 and s.rival_monster_atk_max >= 2500)),
        ),
        (
            "card_id:激流葬|command_type:ACTIVATE",
            lambda s: s.rival_mzone_count >= 2
            and s.rival_mzone_count > s.my_mzone_count
            and not s.has_face_card_on_mzone("青眼の白龍"),
        ),
        (
            "card_id:聖なるバリア －ミラーフォース－|command_type:ACTIVATE",
            lambda s: s.is_battle_phase and s.rival_monster_atk_max >= s.my_lp,
        ),
        (
            "card_id:聖なるバリア －ミラーフォース－|command_type:ACTIVATE",
            lambda s: s.is_battle_phase and s.rival_face_up_atk_monster_count >= 3,
        ),
        (
            "card_id:聖なるバリア －ミラーフォース－|command_type:ACTIVATE",
            lambda s: s.is_battle_phase
            and s.rival_face_up_atk_monster_count >= 2
            and (s.rival_monster_atk_max >= 2000 or s.my_lp <= 4000),
        ),
        (
            "card_id:月の書|command_type:ACTIVATE",
            lambda s: s.rival_mzone_count >= 1 and (s.rival_monster_atk_max >= s.my_lp or s.my_lp <= 1500),
        ),
        (
            "card_id:月の書|command_type:ACTIVATE",
            lambda s: s.is_chaining
            and (
                s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
                or s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－")
            )
            and (s.rival_monster_atk_max >= 2000 or s.rival_mzone_count >= 2 or s.has_face_card_on_mzone("青眼の白龍")),
        ),
        (
            "card_id:月の書|command_type:ACTIVATE",
            lambda s: s.is_chaining
            and s.has_face_card_on_mzone("青眼の白龍")
            and (s.has_rival_card_in_chain_stack("収縮") or s.has_rival_card_in_chain_stack("禁じられた聖槍")),
        ),
        (
            "card_id:月の書|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.is_main1
            and s.has_face_card_on_mzone("白竜の聖騎士")
            and s.rival_mzone_count >= 1
            and (s.rival_monster_atk_max >= 2000 or s.rival_monster_atk_max > s.my_monster_atk_max),
        ),
        (
            "card_id:月の書|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and (
                s.rival_attacking_monster_id == CARD_MAP["アサルトワイバーン"]["id"]
                or s.rival_attacking_monster_id == CARD_MAP["創世の竜騎士"]["id"]
                or s.rival_attacking_monster_atk >= 3000
                or (
                    s.my_mzone_count == 1
                    and s.rival_mzone_count >= 2
                    and s.rival_attacking_monster_atk > s.my_monster_atk_max
                )
                or s.rival_mzone_count >= 3
            ),
        ),
        (
            "card_id:月の書|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.rival_face_up_atk_monster_count >= 1
            and s.my_monster_atk_max < s.rival_monster_atk_max
            and s.my_monster_atk_max >= s.rival_monster_def_min,
        ),
        (
            "card_id:月の書|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.is_main1
            and s.my_lp > 4000
            and (
                (
                    s.has_face_card_on_mzone("センジュ・ゴッド")
                    and not s.has_card_in_chain_stack("センジュ・ゴッド")
                    and not s.has_my_card_used_effect1("センジュ・ゴッド")
                    and not s.has_card_in_hand("白竜の聖騎士")
                )
                or (
                    s.has_face_card_on_mzone("ソニックバード")
                    and not s.has_card_in_chain_stack("ソニックバード")
                    and not s.has_my_card_used_effect1("ソニックバード")
                    and not s.has_card_in_hand("白竜降臨")
                    and not s.has_card_in_hand("高等儀式術")
                )
                or (
                    s.has_face_card_on_mzone("マンジュ・ゴッド")
                    and not s.has_card_in_chain_stack("マンジュ・ゴッド")
                    and not s.has_my_card_used_effect1("マンジュ・ゴッド")
                    and (
                        not s.has_card_in_hand("白竜の聖騎士")
                        or (not s.has_card_in_hand("白竜降臨") and not s.has_card_in_hand("高等儀式術"))
                    )
                )
            ),
        ),
        (
            "card_id:リビングデッドの呼び声|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.is_main1
            and s.has_card_in_grave("青眼の白龍")
            and (s.my_monster_atk_sum + 3000) >= s.rival_lp
            and s.rival_mzone_count == 0,
        ),
        (
            "card_id:リビングデッドの呼び声|command_type:ACTIVATE",
            lambda s: s.is_chaining
            and s.is_last_chain_rival
            and s.has_my_card_targeted_anywhere("リビングデッドの呼び声"),
        ),
        (
            "card_id:リビングデッドの呼び声|command_type:ACTIVATE",
            lambda s: s.is_rival_turn and (s.is_end_phase or s.is_phase_none) and s.has_card_in_grave("青眼の白龍"),
        ),
        (
            "card_id:リビングデッドの呼び声|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.is_main1
            and s.has_card_in_grave("青眼の白龍")
            and (s.my_monster_atk_max < s.rival_monster_atk_max or s.my_mzone_count == 0),
        ),
        (
            "card_id:リビングデッドの呼び声|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and s.my_mzone_count == 0
            and s.has_card_in_grave("青眼の白龍")
            and (s.rival_attacking_monster_atk <= 3000 or s.rival_attacking_monster_atk >= s.my_lp),
        ),
        (
            "card_id:リビングデッドの呼び声|command_type:ACTIVATE",
            lambda s: (
                s.is_my_turn
                and s.is_main1
                and (
                    (s.rival_mzone_count > 0 and s.my_grave_monster_atk_max > s.rival_monster_atk_max)
                    or (s.rival_mzone_count == 0 and s.my_grave_monster_atk_max >= 1900)
                )
                and s.my_mzone_count <= 1
            ),
        ),
        (
            "card_id:リビングデッドの呼び声|command_type:ACTIVATE",
            lambda s: (
                s.is_rival_turn
                and s.is_battle_phase
                and s.my_mzone_count == 0
                and (
                    s.my_lp <= s.rival_attacking_monster_atk or s.rival_attacking_monster_atk >= 2400 or s.my_lp <= 2000
                )
            ),
        ),
        (
            "card_id:リビングデッドの呼び声|command_type:ACTIVATE",
            lambda s: (
                s.is_rival_turn
                and (s.is_end_phase or s.is_phase_none)
                and not s.has_card_in_grave("青眼の白龍")
                and s.my_grave_monster_atk_max >= 1900
                and s.my_mzone_count <= 1
            ),
        ),
        (
            "card_id:銀龍の轟咆|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.is_main1
            and s.rival_lp <= 3000
            and s.rival_mzone_count == 0
            and s.has_card_in_grave("青眼の白龍"),
        ),
        (
            "card_id:銀龍の轟咆|command_type:ACTIVATE",
            lambda s: s.is_rival_turn and (s.is_end_phase or s.is_phase_none) and s.has_card_in_grave("青眼の白龍"),
        ),
        (
            "card_id:銀龍の轟咆|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and s.my_mzone_count == 0
            and s.rival_monster_atk_max <= CARD_MAP["青眼の白龍"]["atk"]
            and s.has_card_in_grave("青眼の白龍"),
        ),
        (
            "card_id:銀龍の轟咆|command_type:ACTIVATE",
            lambda s: s.is_my_turn and s.is_main1 and s.has_card_in_grave("青眼の白龍"),
        ),
        (
            "card_id:銀龍の轟咆|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and s.my_mzone_count == 0
            and s.rival_monster_atk_max >= s.my_lp
            and (
                s.has_card_in_grave("アレキサンドライドラゴン")
                or s.has_card_in_grave("洞窟に潜む竜")
                or s.has_card_in_grave("サファイアドラゴン")
            ),
        ),
        (
            "card_id:銀龍の轟咆|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.is_main2
            and s.my_mzone_count == 0
            and s.my_lp <= 4000
            and s.has_card_in_grave("青眼の白龍"),
        ),
        (
            "card_id:戦線復帰|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and s.my_mzone_count == 0
            and (s.my_lp <= 2000 or s.rival_monster_atk_max >= s.my_lp),
        ),
        (
            "card_id:戦線復帰|command_type:ACTIVATE",
            lambda s: s.is_rival_turn and (s.is_end_phase or s.is_phase_none) and s.has_card_in_grave("青眼の白龍"),
        ),
        (
            "card_id:戦線復帰|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and (s.is_end_phase or s.is_phase_none)
            and s.has_card_in_grave("創世の竜騎士")
            and s.has_card_in_grave("青眼の白龍")
            and (s.my_hand_count >= 1 or s.rival_mzone_count == 0),
        ),
        (
            "card_id:戦線復帰|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and (s.is_end_phase or s.is_phase_none)
            and s.has_card_in_grave("アサルトワイバーン")
            and (s.has_card_in_grave("青眼の白龍") or s.has_card_in_hand("青眼の白龍"))
            and (
                s.rival_mzone_count == 0 or s.rival_monster_atk_min < 1800 or s.rival_face_up_def_monster_def_min < 1800
            ),
        ),
        (
            "card_id:戦線復帰|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and s.my_mzone_count == 0
            and s.has_card_in_grave("青眼の白龍")
            and s.rival_attacking_monster_atk >= 2000,
        ),
        (
            "card_id:強化蘇生|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and s.my_mzone_count == 0
            and s.rival_monster_atk_max >= s.my_lp - 1500,
        ),
        (
            "card_id:強化蘇生|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.is_main1
            and s.has_card_in_grave("白竜の聖騎士")
            and (s.has_card_in_deck("青眼の白龍") or s.has_card_in_hand("青眼の白龍")),
        ),
        (
            "card_id:強化蘇生|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.is_main1
            and s.has_card_in_grave("創世の竜騎士")
            and s.has_card_in_grave("青眼の白龍")
            and s.my_hand_count >= 1,
        ),
        (
            "card_id:強化蘇生|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and (s.is_end_phase or s.is_phase_none)
            and s.has_card_in_grave("白竜の聖騎士")
            and (s.has_card_in_deck("青眼の白龍") or s.has_card_in_hand("青眼の白龍")),
        ),
        (
            "card_id:強化蘇生|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and (s.is_end_phase or s.is_phase_none)
            and s.my_mzone_count <= 1
            and (
                s.has_card_in_grave("創世の竜騎士")
                or s.has_card_in_grave("アサルトワイバーン")
                or s.has_card_in_grave("アレキサンドライドラゴン")
            ),
        ),
        (
            "card_id:強化蘇生|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and s.my_mzone_count == 0
            and (
                s.has_card_in_grave("仮面竜")
                or s.has_card_in_grave("ボマー・ドラゴン")
                or (
                    s.has_card_in_grave("洞窟に潜む竜")
                    and s.rival_attacking_monster_atk < CARD_MAP["洞窟に潜む竜"]["def"] + 100
                )
            ),
        ),
        (
            "card_id:収縮|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and (s.my_lp <= 2000 or s.rival_monster_atk_max >= s.my_lp)
            and s.rival_monster_atk_max > 0,
        ),
        (
            "card_id:収縮|command_type:ACTIVATE",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and s.is_dmg_before_calc
            and s.my_attacked_monster_atk < s.rival_attacking_monster_atk
            and s.my_attacked_monster_atk >= (s.rival_attacking_monster_atk // 2),
        ),
        (
            "card_id:収縮|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.is_battle_phase
            and s.is_dmg_before_calc
            and s.my_attacking_monster_atk < s.rival_attacked_monster_atk
            and s.my_attacking_monster_atk >= (s.rival_attacked_monster_atk // 2),
        ),
        ("card_id:マンジュ・ゴッド|command_type:ACTIVATE", lambda s: s.my_deck_count >= 2),
        ("card_id:センジュ・ゴッド|command_type:ACTIVATE", lambda s: s.my_deck_count >= 2),
        ("card_id:ソニックバード|command_type:ACTIVATE", lambda s: s.my_deck_count >= 2),
        (
            "card_id:アサルトワイバーン|command_type:ACTIVATE",
            lambda s: s.my_grave_count >= 1 or s.my_hand_monster_count >= 1,
        ),
        ("card_id:仮面竜|command_type:ACTIVATE", lambda s: s.my_deck_count >= 2),
        (
            "card_id:仮面竜|command_type:ACTIVATE",
            lambda s: s.my_deck_count == 1 and s.rival_monster_atk_max >= s.my_lp and s.my_mzone_count == 0,
        ),
        ("card_id:コドモドラゴン|command_type:ACTIVATE", lambda s: s.is_rival_turn or s.is_main2),
        (
            "card_id:コドモドラゴン|command_type:ACTIVATE",
            lambda s: s.is_my_turn
            and s.is_main1
            and s.has_card_in_hand("青眼の白龍")
            and s.rival_lp > s.my_monster_atk_sum,
        ),
        (
            "card_id:コドモドラゴン|command_type:ACTIVATE",
            lambda s: s.is_my_turn and s.is_main1 and s.my_mzone_count == 0,
        ),
        ("card_id:創世の竜騎士|command_type:ACTIVATE", lambda s: s.my_deck_count >= 2),
        ("card_id:None|command_type:PASS", lambda s: True),
        ("card_id:サイクロン|command_type:ACTIVATE", lambda s: True),
        ("card_id:砂塵の大竜巻|command_type:ACTIVATE", lambda s: True),
        ("card_id:禁じられた聖槍|command_type:ACTIVATE", lambda s: True),
        ("card_id:激流葬|command_type:ACTIVATE", lambda s: True),
        ("card_id:聖なるバリア －ミラーフォース－|command_type:ACTIVATE", lambda s: True),
        ("card_id:月の書|command_type:ACTIVATE", lambda s: True),
        ("card_id:リビングデッドの呼び声|command_type:ACTIVATE", lambda s: True),
        ("card_id:銀龍の轟咆|command_type:ACTIVATE", lambda s: True),
        ("card_id:戦線復帰|command_type:ACTIVATE", lambda s: True),
        ("card_id:強化蘇生|command_type:ACTIVATE", lambda s: True),
        ("card_id:収縮|command_type:ACTIVATE", lambda s: True),
        ("card_id:マンジュ・ゴッド|command_type:ACTIVATE", lambda s: True),
        ("card_id:センジュ・ゴッド|command_type:ACTIVATE", lambda s: True),
        ("card_id:ソニックバード|command_type:ACTIVATE", lambda s: True),
        ("card_id:アサルトワイバーン|command_type:ACTIVATE", lambda s: True),
        ("card_id:仮面竜|command_type:ACTIVATE", lambda s: True),
        ("card_id:コドモドラゴン|command_type:ACTIVATE", lambda s: True),
        ("card_id:創世の竜騎士|command_type:ACTIVATE", lambda s: True),
    ]

    # 選択可能な行動を評価
    command_scores: list[float] = []

    for target_command in selectable_commands:
        target_command_score: float = float("-inf")
        target_command_identifier: str | None = None

        # 識別子を特定
        for command_identifier, command_condition in OPTIONS.items():
            if all(getattr(target_command, attr, None) == val for attr, val in command_condition.items()):
                target_command_identifier = command_identifier
                break

        # 例外処理
        if not target_command_identifier:
            write_debug_log(
                selectable_commands, selection_type, selection_id, f"Unidentified command: {target_command}"
            )

        # ランキングで評価
        for rank_index, (rank_identifier, rank_condition) in enumerate(ranking):
            rank_score = len(ranking) - rank_index

            if rank_identifier == target_command_identifier and rank_condition(situation):
                target_command_score = rank_score
                break

        # 評価結果を保存
        command_scores.append(target_command_score)

    # 最良行動を抽出
    max_score = max(command_scores)
    candidate_indices = [i for i, score in enumerate(command_scores) if score == max_score]

    # 最良行動を選択
    best_command_index = candidate_indices[0]

    if len(candidate_indices) > 1:
        # `pos_id`選択 (`MAGIC` > `HAND`)
        magic_positions = [PosId.MAGIC_L_L, PosId.MAGIC_L, PosId.MAGIC_C, PosId.MAGIC_R, PosId.MAGIC_R_R]
        random.shuffle(magic_positions)

        for priority_pos in magic_positions + [PosId.HAND]:
            if filtered_indices := [i for i in candidate_indices if selectable_commands[i].pos_id == priority_pos]:
                candidate_indices = filtered_indices
                break

        # ランダム選択 (`card_index`, `table_index`)
        def command_signature(i: int):
            return {
                key: val
                for key, val in vars(selectable_commands[i]).items()
                if key not in ("card_index", "table_index")
            }

        if any(command_signature(i) != command_signature(candidate_indices[0]) for i in candidate_indices):
            write_debug_log(
                selectable_commands,
                selection_type,
                selection_id,
                f"Ambiguous commands: {[selectable_commands[i] for i in candidate_indices]}",
            )

        best_command_index = random.choice(candidate_indices)

    return best_command_index
