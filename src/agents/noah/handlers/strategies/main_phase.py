# 有利な時は、相手のミラーフォースや激流葬を警戒してモンスターを出さない
# 自分の場に青眼の白龍が出ている場合は、相手の伏せカードを、大嵐で破壊してもいいかも（青眼を大事にする方針。）
import random
from typing import Callable

from ygo.constants.enums import CommandType, PosId
from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.main_phase import OPTIONS
from ..utils import CARD_MAP, Situation, write_debug_log


def select_main_phase(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """メインフェイズ (SelectionType:1) & 値無し (SelectionId:-1)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        ("card_id:強欲な壺|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        (
            "card_id:ライトニング・ボルテックス|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_mzone_count >= 3,
        ),
        (
            "card_id:大嵐|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_szone_count >= 3
            and s.my_szone_count <= 1
            and not (s.has_card_on_szone("聖なるバリア －ミラーフォース－") or s.has_card_on_szone("激流葬")),
        ),
        (
            "card_id:ライトニング・ボルテックス|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_mzone_count >= 2 and s.rival_monster_atk_max > s.my_monster_atk_max,
        ),
        (
            "card_id:大嵐|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_szone_count >= 2 and s.my_szone_count == 0,
        ),
        (
            "card_id:サイクロン|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_szone_count >= 1
            and (
                (
                    s.has_card_in_hand("白竜降臨")
                    and s.has_card_in_hand("白竜の聖騎士")
                    and (
                        s.my_hand_monster_count
                        + s.my_mzone_count
                        - s.my_blue_eyes_hand_count
                        - s.my_blue_eyes_mzone_count
                    )
                    >= 2
                )
                or (
                    s.has_card_in_hand("高等儀式術")
                    and s.has_card_in_hand("白竜の聖騎士")
                    and any(
                        s.has_card_in_deck(n)
                        for n in ["アレキサンドライドラゴン", "洞窟に潜む竜", "サファイアドラゴン"]
                    )
                )
            ),
        ),
        (
            "card_id:砂塵の大竜巻|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_szone_count >= 1
            and (
                (
                    s.has_card_in_hand("白竜降臨")
                    and s.has_card_in_hand("白竜の聖騎士")
                    and (
                        s.my_hand_monster_count
                        + s.my_mzone_count
                        - s.my_blue_eyes_hand_count
                        - s.my_blue_eyes_mzone_count
                    )
                    >= 2
                )
                or (
                    s.has_card_in_hand("高等儀式術")
                    and s.has_card_in_hand("白竜の聖騎士")
                    and any(
                        s.has_card_in_deck(n)
                        for n in ["アレキサンドライドラゴン", "洞窟に潜む竜", "サファイアドラゴン"]
                    )
                )
            ),
        ),
        (
            "card_id:大嵐|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_szone_count >= 2
            and s.my_szone_count <= 1
            and not (s.has_card_on_szone("聖なるバリア －ミラーフォース－") or s.has_card_on_szone("激流葬"))
            and (
                (
                    s.has_card_in_hand("白竜降臨")
                    and s.has_card_in_hand("白竜の聖騎士")
                    and (
                        s.my_hand_monster_count
                        + s.my_mzone_count
                        - s.my_blue_eyes_hand_count
                        - s.my_blue_eyes_mzone_count
                    )
                    >= 2
                )
                or (
                    s.has_card_in_hand("高等儀式術")
                    and s.has_card_in_hand("白竜の聖騎士")
                    and any(
                        s.has_card_in_deck(n)
                        for n in ["アレキサンドライドラゴン", "洞窟に潜む竜", "サファイアドラゴン"]
                    )
                )
            ),
        ),
        (
            "card_id:ライトニング・ボルテックス|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_monster_atk_max >= 3000 and s.my_monster_atk_max < 3000,
        ),
        (
            "card_id:サイクロン|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_szone_count >= 1 and (s.my_lp <= 3000 or s.rival_hand_count <= 1),
        ),
        (
            "card_id:砂塵の大竜巻|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_szone_count >= 1 and (s.my_lp <= 3000 or s.rival_hand_count <= 1),
        ),
        (
            "card_id:大嵐|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_szone_count >= 1
            and s.my_lp <= 3000
            and (
                s.my_szone_count == 0
                or (
                    s.my_szone_count == 1
                    and not (s.has_card_on_szone("聖なるバリア －ミラーフォース－") or s.has_card_on_szone("激流葬"))
                )
            ),
        ),
        (
            "card_id:ライトニング・ボルテックス|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.rival_mzone_count >= 1 and s.my_lp <= 3000 and s.rival_monster_atk_max > s.my_monster_atk_max,
        ),
        (
            "card_id:センジュ・ゴッド|command_type:SUMMON|phase:NO_VALUE",
            lambda s: (not s.has_card_in_hand("白竜の聖騎士") and s.has_card_in_deck("白竜の聖騎士"))
            and (
                s.has_card_in_hand("白竜降臨")
                or (
                    s.has_card_in_hand("高等儀式術")
                    and any(
                        s.has_card_in_deck(n)
                        for n in ["アレキサンドライドラゴン", "洞窟に潜む竜", "サファイアドラゴン"]
                    )
                )
            ),
        ),
        (
            "card_id:マンジュ・ゴッド|command_type:SUMMON|phase:NO_VALUE",
            lambda s: (not s.has_card_in_hand("白竜の聖騎士") and s.has_card_in_deck("白竜の聖騎士"))
            and (
                s.has_card_in_hand("白竜降臨")
                or (
                    s.has_card_in_hand("高等儀式術")
                    and any(
                        s.has_card_in_deck(n)
                        for n in ["アレキサンドライドラゴン", "洞窟に潜む竜", "サファイアドラゴン"]
                    )
                )
            ),
        ),
        (
            "card_id:ソニックバード|command_type:SUMMON|phase:NO_VALUE",
            lambda s: (
                not (
                    s.has_card_in_hand("高等儀式術")
                    and any(
                        s.has_card_in_deck(n)
                        for n in ["アレキサンドライドラゴン", "洞窟に潜む竜", "サファイアドラゴン"]
                    )
                )
                and (s.has_card_in_deck("白竜降臨") or s.has_card_in_deck("高等儀式術"))
            )
            and s.has_card_in_hand("白竜の聖騎士"),
        ),
        (
            "card_id:マンジュ・ゴッド|command_type:SUMMON|phase:NO_VALUE",
            lambda s: (
                not (
                    s.has_card_in_hand("高等儀式術")
                    and any(
                        s.has_card_in_deck(n)
                        for n in ["アレキサンドライドラゴン", "洞窟に潜む竜", "サファイアドラゴン"]
                    )
                )
                and (s.has_card_in_deck("白竜降臨") or s.has_card_in_deck("高等儀式術"))
            )
            and s.has_card_in_hand("白竜の聖騎士"),
        ),
        (
            "card_id:高等儀式術|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: True,
        ),
        (
            "card_id:白竜降臨|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.has_card_in_hand("白竜の聖騎士")
            and (
                (
                    fodder_count := (
                        s.my_hand_monster_count
                        + s.my_mzone_count
                        - s.my_blue_eyes_mzone_count
                        - (
                            1
                            if (
                                s.has_card_in_hand("アレキサンドライドラゴン")
                                or s.has_face_card_on_mzone("アレキサンドライドラゴン")
                            )
                            and CARD_MAP["白竜の聖騎士"]["atk"]
                            <= s.rival_monster_atk_max
                            < CARD_MAP["アレキサンドライドラゴン"]["atk"]
                            else 0
                        )
                        - (
                            1
                            if (s.has_card_in_hand("ボマー・ドラゴン") or s.has_face_card_on_mzone("ボマー・ドラゴン"))
                            and s.has_rival_card_on_mzone("青眼の白龍")
                            and s.rival_monster_atk_max >= s.my_monster_atk_max
                            else 0
                        )
                        - (
                            1
                            if (
                                s.has_card_in_hand("アサルトワイバーン")
                                or s.has_face_card_on_mzone("アサルトワイバーン")
                            )
                            and s.rival_mzone_count >= 1
                            and (
                                s.rival_monster_atk_min <= CARD_MAP["アサルトワイバーン"]["atk"]
                                or s.rival_monster_def_min <= CARD_MAP["アサルトワイバーン"]["atk"]
                            )
                            and (
                                s.has_card_in_hand("青眼の白龍")
                                or s.has_card_in_grave("青眼の白龍")
                                or s.has_card_in_hand("白竜の聖騎士")
                                or s.has_card_in_grave("白竜の聖騎士")
                            )
                            else 0
                        )
                        - (
                            1
                            if (s.has_card_in_hand("創世の竜騎士") or s.has_face_card_on_mzone("創世の竜騎士"))
                            and (
                                s.has_card_in_grave("青眼の白龍")
                                or (
                                    s.rival_mzone_count >= 1
                                    and (
                                        s.rival_monster_atk_min <= CARD_MAP["創世の竜騎士"]["atk"]
                                        or s.rival_monster_def_min <= CARD_MAP["創世の竜騎士"]["atk"]
                                    )
                                )
                            )
                            and s.my_hand_count
                            > sum(
                                s.has_card_in_hand(n)
                                for n in [
                                    "強欲な壺",
                                    "死者蘇生",
                                    "大嵐",
                                    "ライトニング・ボルテックス",
                                    "早すぎた埋葬",
                                    "銀龍の轟咆",
                                    "戦線復帰",
                                    "強化蘇生",
                                    "リビングデッドの呼び声",
                                    "聖なるバリア －ミラーフォース－",
                                    "激流葬",
                                ]
                            )
                            else 0
                        )
                    )
                )
                >= 2
            )
            and (
                (fodder_count - s.my_blue_eyes_hand_count >= 2)
                or (
                    s.my_blue_eyes_hand_count >= 1
                    and (
                        s.has_card_in_hand("死者蘇生")
                        or s.has_card_in_hand("銀龍の轟咆")
                        or s.has_card_on_szone("銀龍の轟咆")
                        or s.has_card_in_hand("リビングデッドの呼び声")
                        or s.has_card_on_szone("リビングデッドの呼び声")
                        or s.has_card_in_hand("戦線復帰")
                        or s.has_card_on_szone("戦線復帰")
                        or s.has_card_in_hand("強化蘇生")
                        or s.has_card_on_szone("強化蘇生")
                    )
                )
                or (s.my_blue_eyes_hand_count >= 2)
            ),
        ),
        (
            "card_id:白竜の聖騎士|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.is_first_turn or s.is_main2 or s.has_my_card_used_effect1("コドモドラゴン"),
        ),
        (
            "card_id:創世の竜騎士|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.my_hand_count
            > sum(
                s.has_card_in_hand(n)
                for n in [
                    "強欲な壺",
                    "死者蘇生",
                    "大嵐",
                    "ライトニング・ボルテックス",
                    "早すぎた埋葬",
                    "銀龍の轟咆",
                    "戦線復帰",
                    "強化蘇生",
                    "リビングデッドの呼び声",
                    "聖なるバリア －ミラーフォース－",
                    "激流葬",
                ]
            ),
        ),
        (
            "card_id:死者蘇生|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.is_main1
            and (
                s.has_card_in_grave("青眼の白龍")
                or s.has_rival_card_in_grave("青眼の白龍")
                or (s.my_monster_atk_sum + max(s.my_grave_monster_atk_max, s.rival_grave_monster_atk_max) >= s.rival_lp)
                or (s.rival_monster_atk_sum >= s.my_lp)
                or (
                    s.has_card_in_grave("白竜の聖騎士")
                    and (s.has_card_in_deck("青眼の白龍") or s.has_card_in_hand("青眼の白龍"))
                )
            ),
        ),
        (
            "card_id:銀龍の轟咆|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.is_main1
            and (
                s.has_card_in_grave("青眼の白龍")
                or (
                    any(
                        s.has_card_in_grave(n)
                        for n in ["アレキサンドライドラゴン", "サファイアドラゴン", "洞窟に潜む竜"]
                    )
                    and (
                        (
                            s.has_card_in_grave("アレキサンドライドラゴン")
                            and s.my_monster_atk_sum + CARD_MAP["アレキサンドライドラゴン"]["atk"] >= s.rival_lp
                        )
                        or (
                            s.has_card_in_grave("サファイアドラゴン")
                            and s.my_monster_atk_sum + CARD_MAP["サファイアドラゴン"]["atk"] >= s.rival_lp
                        )
                        or (
                            s.has_card_in_grave("洞窟に潜む竜")
                            and s.my_monster_atk_sum + CARD_MAP["洞窟に潜む竜"]["atk"] >= s.rival_lp
                        )
                        or (s.rival_monster_atk_sum >= s.my_lp)
                    )
                )
            ),
        ),
        (
            "card_id:早すぎた埋葬|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.is_main1
            and s.my_lp > 800
            and (
                s.has_card_in_grave("青眼の白龍")
                or (s.my_monster_atk_sum + s.my_grave_monster_atk_max >= s.rival_lp)
                or (s.rival_monster_atk_sum >= s.my_lp)
                or (
                    s.has_card_in_grave("白竜の聖騎士")
                    and (s.has_card_in_deck("青眼の白龍") or s.has_card_in_hand("青眼の白龍"))
                )
            ),
        ),
        (
            "card_id:リビングデッドの呼び声|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.is_main1
            and (
                s.has_card_in_grave("青眼の白龍")
                or (s.my_monster_atk_sum + s.my_grave_monster_atk_max >= s.rival_lp)
                or (s.rival_monster_atk_sum >= s.my_lp)
                or (s.my_grave_monster_atk_max >= 1900 and (s.my_szone_count >= 3 or s.rival_mzone_count >= 2))
                or (
                    s.has_card_in_grave("白竜の聖騎士")
                    and (s.has_card_in_deck("青眼の白龍") or s.has_card_in_hand("青眼の白龍"))
                )
            ),
        ),
        (
            "card_id:強化蘇生|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.is_main1
            and (
                (
                    s.has_card_in_grave("白竜の聖騎士")
                    and (s.has_card_in_deck("青眼の白龍") or s.has_card_in_hand("青眼の白龍"))
                )
                or (s.has_card_in_grave("創世の竜騎士") and s.has_card_in_grave("青眼の白龍") and s.my_hand_count >= 1)
                or (
                    s.has_card_in_grave("アサルトワイバーン")
                    and (s.has_card_in_grave("青眼の白龍") or s.has_card_in_hand("青眼の白龍"))
                    and (s.rival_monster_atk_min <= CARD_MAP["アサルトワイバーン"]["atk"])
                )
                or (
                    s.has_card_in_grave("アレキサンドライドラゴン")
                    and s.my_monster_atk_sum + CARD_MAP["アレキサンドライドラゴン"]["atk"] + 100 >= s.rival_lp
                )
                or (
                    s.has_card_in_grave("サファイアドラゴン")
                    and s.my_monster_atk_sum + CARD_MAP["サファイアドラゴン"]["atk"] + 100 >= s.rival_lp
                )
                or (s.rival_monster_atk_sum >= s.my_lp)
                or (s.my_grave_monster_atk_max >= 1900 and (s.my_szone_count >= 3 or s.rival_mzone_count >= 2))
            ),
        ),
        (
            "card_id:月の書|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.is_main1
            and s.my_lp > 4000
            and (
                (
                    s.has_face_card_on_mzone("センジュ・ゴッド")
                    and not s.has_my_card_used_effect1("センジュ・ゴッド")
                    and not s.has_card_in_hand("白竜の聖騎士")
                )
                or (
                    s.has_face_card_on_mzone("ソニックバード")
                    and not s.has_my_card_used_effect1("ソニックバード")
                    and not (s.has_card_in_hand("白竜降臨") or s.has_card_in_hand("高等儀式術"))
                )
                or (
                    s.has_face_card_on_mzone("マンジュ・ゴッド")
                    and not s.has_my_card_used_effect1("マンジュ・ゴッド")
                    and (
                        not s.has_card_in_hand("白竜の聖騎士")
                        or not (s.has_card_in_hand("白竜降臨") or s.has_card_in_hand("高等儀式術"))
                    )
                )
            ),
        ),
        (
            "card_id:アレキサンドライドラゴン|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.rival_mzone_count == 0
            and s.my_monster_atk_sum + CARD_MAP["アレキサンドライドラゴン"]["atk"] >= s.rival_lp,
        ),
        (
            "card_id:サファイアドラゴン|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.rival_mzone_count == 0
            and s.my_monster_atk_sum + CARD_MAP["サファイアドラゴン"]["atk"] >= s.rival_lp,
        ),
        (
            "card_id:創世の竜騎士|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.rival_mzone_count == 0 and s.my_monster_atk_sum + CARD_MAP["創世の竜騎士"]["atk"] >= s.rival_lp,
        ),
        (
            "card_id:アサルトワイバーン|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.rival_mzone_count == 0
            and s.my_monster_atk_sum + CARD_MAP["アサルトワイバーン"]["atk"] >= s.rival_lp,
        ),
        (
            "card_id:仮面竜|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.rival_mzone_count == 0 and s.my_monster_atk_sum + CARD_MAP["仮面竜"]["atk"] >= s.rival_lp,
        ),
        (
            "card_id:洞窟に潜む竜|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.rival_mzone_count == 0
            and s.rival_szone_count == 0
            and s.my_monster_atk_sum + CARD_MAP["洞窟に潜む竜"]["atk"] >= s.rival_lp,
        ),
        (
            "card_id:コドモドラゴン|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.rival_mzone_count == 0
            and s.rival_szone_count == 0
            and s.my_monster_atk_sum + CARD_MAP["コドモドラゴン"]["atk"] >= s.rival_lp,
        ),
        (
            "card_id:青眼の白龍|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.rival_mzone_count == 0
            and s.my_mzone_count >= 2
            and s.my_monster_atk_sum < s.rival_lp
            and (s.my_monster_atk_sum - s.my_monster_atk_min_2sum + 3000) >= s.rival_lp,
        ),
        (
            "card_id:創世の竜騎士|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.has_card_in_grave("青眼の白龍")
            and (
                s.my_hand_count
                - sum(
                    s.has_card_in_hand(n)
                    for n in [
                        "強欲な壺",
                        "死者蘇生",
                        "大嵐",
                        "ライトニング・ボルテックス",
                        "早すぎた埋葬",
                        "銀龍の轟咆",
                        "戦線復帰",
                        "強化蘇生",
                        "リビングデッドの呼び声",
                        "聖なるバリア －ミラーフォース－",
                        "激流葬",
                    ]
                )
            )
            >= 2,
        ),
        (
            "card_id:アサルトワイバーン|command_type:SUMMON|phase:NO_VALUE",
            lambda s: (
                (
                    s.rival_face_up_atk_monster_count >= 1
                    and s.rival_monster_atk_min <= CARD_MAP["アサルトワイバーン"]["atk"]
                )
                or (
                    s.rival_face_up_def_monster_count >= 1
                    and s.rival_monster_def_min <= CARD_MAP["アサルトワイバーン"]["atk"]
                )
            )
            and (
                s.has_card_in_hand("青眼の白龍")
                or s.has_card_in_grave("青眼の白龍")
                or s.has_card_in_grave("白竜の聖騎士")
            ),
        ),
        (
            "card_id:創世の竜騎士|command_type:SUMMON|phase:NO_VALUE",
            lambda s: (
                (s.rival_face_up_atk_monster_count >= 1 and s.rival_monster_atk_min < CARD_MAP["創世の竜騎士"]["atk"])
                or (
                    s.rival_face_up_def_monster_count >= 1 and s.rival_monster_def_min < CARD_MAP["創世の竜騎士"]["atk"]
                )
            )
            and s.has_card_in_deck("青眼の白龍"),
        ),
        (
            "card_id:ボマー・ドラゴン|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.rival_monster_atk_max >= 3000
            or max(s.my_hand_max_normal_summon_atk, s.my_monster_atk_max) < s.rival_monster_atk_max,
        ),
        (
            "card_id:仮面竜|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.rival_monster_atk_max >= 3000 and s.has_card_in_deck("ボマー・ドラゴン") and s.is_main1,
        ),
        (
            "card_id:センジュ・ゴッド|command_type:SUMMON|phase:NO_VALUE",
            lambda s: (not s.has_card_in_hand("白竜の聖騎士") and s.has_card_in_deck("白竜の聖騎士"))
            and (
                not (
                    s.has_card_in_hand("高等儀式術")
                    and any(
                        s.has_card_in_deck(n)
                        for n in ["アレキサンドライドラゴン", "洞窟に潜む竜", "サファイアドラゴン"]
                    )
                )
                and (s.has_card_in_deck("白竜降臨") or s.has_card_in_deck("高等儀式術"))
            ),
        ),
        (
            "card_id:ソニックバード|command_type:SUMMON|phase:NO_VALUE",
            lambda s: (not s.has_card_in_hand("白竜の聖騎士") and s.has_card_in_deck("白竜の聖騎士"))
            and (
                not (
                    s.has_card_in_hand("高等儀式術")
                    and any(
                        s.has_card_in_deck(n)
                        for n in ["アレキサンドライドラゴン", "洞窟に潜む竜", "サファイアドラゴン"]
                    )
                )
                and (s.has_card_in_deck("白竜降臨") or s.has_card_in_deck("高等儀式術"))
            ),
        ),
        (
            "card_id:マンジュ・ゴッド|command_type:SUMMON|phase:NO_VALUE",
            lambda s: (not s.has_card_in_hand("白竜の聖騎士") and s.has_card_in_deck("白竜の聖騎士"))
            and (
                not (
                    s.has_card_in_hand("高等儀式術")
                    and any(
                        s.has_card_in_deck(n)
                        for n in ["アレキサンドライドラゴン", "洞窟に潜む竜", "サファイアドラゴン"]
                    )
                )
                and (s.has_card_in_deck("白竜降臨") or s.has_card_in_deck("高等儀式術"))
            ),
        ),
        (
            "card_id:仮面竜|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.is_first_turn
            and (s.my_mzone_count == 0 or (s.my_mzone_count == 1 and s.has_card_on_mzone("白竜の聖騎士"))),
        ),
        (
            "card_id:マンジュ・ゴッド|command_type:REVERSE|phase:NO_VALUE",
            lambda s: s.is_main1
            and (
                not s.has_my_card_used_effect1("マンジュ・ゴッド")
                or (
                    s.rival_mzone_count == 0
                    or s.rival_monster_atk_min < CARD_MAP["マンジュ・ゴッド"]["atk"]
                    or (s.my_monster_atk_max >= s.rival_monster_atk_max and s.my_mzone_count > s.rival_mzone_count)
                )
            ),
        ),
        (
            "card_id:青眼の白龍|command_type:REVERSE|phase:NO_VALUE",
            lambda s: s.is_main1
            and (
                s.rival_mzone_count == 0
                or s.rival_monster_atk_min < CARD_MAP["青眼の白龍"]["atk"]
                or (s.my_monster_atk_max >= s.rival_monster_atk_max and s.my_mzone_count > s.rival_mzone_count)
            ),
        ),
        (
            "card_id:青眼の白龍|command_type:TURN_ATK|phase:NO_VALUE",
            lambda s: s.is_main1
            and (
                s.rival_mzone_count == 0
                or s.rival_monster_atk_min < CARD_MAP["青眼の白龍"]["atk"]
                or (s.my_monster_atk_max >= s.rival_monster_atk_max and s.my_mzone_count > s.rival_mzone_count)
            ),
        ),
        (
            "card_id:白竜の聖騎士|command_type:TURN_ATK|phase:NO_VALUE",
            lambda s: s.is_main1
            and (
                s.rival_mzone_count == 0
                or s.rival_monster_atk_min < CARD_MAP["白竜の聖騎士"]["atk"]
                or s.rival_face_down_monster_count >= 1
                or (s.my_monster_atk_max >= s.rival_monster_atk_max and s.my_mzone_count > s.rival_mzone_count)
            ),
        ),
        (
            "card_id:アレキサンドライドラゴン|command_type:TURN_ATK|phase:NO_VALUE",
            lambda s: s.is_main1
            and (
                s.rival_mzone_count == 0
                or s.rival_monster_atk_min < CARD_MAP["アレキサンドライドラゴン"]["atk"]
                or (s.my_monster_atk_max >= s.rival_monster_atk_max and s.my_mzone_count > s.rival_mzone_count)
            ),
        ),
        (
            "card_id:ボマー・ドラゴン|command_type:TURN_ATK|phase:NO_VALUE",
            lambda s: s.is_main1
            and s.rival_mzone_count >= 1
            and (
                s.rival_monster_atk_max > max(s.my_monster_atk_max, s.my_hand_max_normal_summon_atk)
                or s.rival_mzone_count > s.my_mzone_count
            ),
        ),
        (
            "card_id:洞窟に潜む竜|command_type:TURN_ATK|phase:NO_VALUE",
            lambda s: s.is_main1
            and s.rival_szone_count == 0
            and (
                s.rival_mzone_count == 0
                or (s.my_monster_atk_max >= s.rival_monster_atk_max and s.my_mzone_count > s.rival_mzone_count)
            ),
        ),
        (
            "card_id:コドモドラゴン|command_type:REVERSE|phase:NO_VALUE",
            lambda s: s.is_main1
            and s.rival_szone_count == 0
            and (
                s.rival_mzone_count == 0
                or (s.my_monster_atk_max >= s.rival_monster_atk_max and s.my_mzone_count > s.rival_mzone_count)
            ),
        ),
        (
            "card_id:洞窟に潜む竜|command_type:SET_MONST|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and not s.has_rival_card_on_mzone("白竜の聖騎士")
            and (s.is_first_turn or s.is_main2 or s.rival_monster_atk_min > s.my_hand_max_normal_summon_atk),
        ),
        (
            "card_id:コドモドラゴン|command_type:SET_MONST|phase:NO_VALUE",
            lambda s: s.my_blue_eyes_hand_count >= 1
            and (s.my_mzone_count == 0 or (s.rival_monster_atk_sum > s.my_monster_atk_sum))
            and (s.is_first_turn or s.is_main2 or s.rival_monster_atk_min > s.my_hand_max_normal_summon_atk),
        ),
        (
            "card_id:創世の竜騎士|command_type:SET_MONST|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and s.my_lp <= s.rival_monster_atk_sum
            and not (
                s.my_hand_max_normal_summon_atk >= s.rival_monster_atk_min
                or s.has_card_in_hand("ボマー・ドラゴン")
                or (
                    (s.has_card_in_hand("収縮") or s.has_card_on_szone("収縮"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min / 2)
                )
                or (
                    (s.has_card_in_hand("禁じられた聖槍") or s.has_card_on_szone("禁じられた聖槍"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min - 800)
                )
            ),
        ),
        (
            "card_id:サファイアドラゴン|command_type:SET_MONST|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and s.my_lp <= s.rival_monster_atk_sum
            and not (
                s.my_hand_max_normal_summon_atk >= s.rival_monster_atk_min
                or s.has_card_in_hand("ボマー・ドラゴン")
                or (
                    (s.has_card_in_hand("収縮") or s.has_card_on_szone("収縮"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min / 2)
                )
                or (
                    (s.has_card_in_hand("禁じられた聖槍") or s.has_card_on_szone("禁じられた聖槍"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min - 800)
                )
            ),
        ),
        (
            "card_id:仮面竜|command_type:SET_MONST|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and s.my_lp <= s.rival_monster_atk_sum
            and not (
                s.my_hand_max_normal_summon_atk >= s.rival_monster_atk_min
                or s.has_card_in_hand("ボマー・ドラゴン")
                or (
                    (s.has_card_in_hand("収縮") or s.has_card_on_szone("収縮"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min / 2)
                )
                or (
                    (s.has_card_in_hand("禁じられた聖槍") or s.has_card_on_szone("禁じられた聖槍"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min - 800)
                )
            ),
        ),
        (
            "card_id:マンジュ・ゴッド|command_type:SET_MONST|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and s.my_lp <= s.rival_monster_atk_sum
            and not (
                s.my_hand_max_normal_summon_atk >= s.rival_monster_atk_min
                or s.has_card_in_hand("ボマー・ドラゴン")
                or (
                    (s.has_card_in_hand("収縮") or s.has_card_on_szone("収縮"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min / 2)
                )
                or (
                    (s.has_card_in_hand("禁じられた聖槍") or s.has_card_on_szone("禁じられた聖槍"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min - 800)
                )
            ),
        ),
        (
            "card_id:センジュ・ゴッド|command_type:SET_MONST|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and s.my_lp <= s.rival_monster_atk_sum
            and not (
                s.my_hand_max_normal_summon_atk >= s.rival_monster_atk_min
                or s.has_card_in_hand("ボマー・ドラゴン")
                or (
                    (s.has_card_in_hand("収縮") or s.has_card_on_szone("収縮"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min / 2)
                )
                or (
                    (s.has_card_in_hand("禁じられた聖槍") or s.has_card_on_szone("禁じられた聖槍"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min - 800)
                )
            ),
        ),
        (
            "card_id:ソニックバード|command_type:SET_MONST|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and s.my_lp <= s.rival_monster_atk_sum
            and not (
                s.my_hand_max_normal_summon_atk >= s.rival_monster_atk_min
                or s.has_card_in_hand("ボマー・ドラゴン")
                or (
                    (s.has_card_in_hand("収縮") or s.has_card_on_szone("収縮"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min / 2)
                )
                or (
                    (s.has_card_in_hand("禁じられた聖槍") or s.has_card_on_szone("禁じられた聖槍"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min - 800)
                )
            ),
        ),
        (
            "card_id:アサルトワイバーン|command_type:SET_MONST|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and s.my_lp <= s.rival_monster_atk_sum
            and not (
                s.my_hand_max_normal_summon_atk >= s.rival_monster_atk_min
                or s.has_card_in_hand("ボマー・ドラゴン")
                or (
                    (s.has_card_in_hand("収縮") or s.has_card_on_szone("収縮"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min / 2)
                )
                or (
                    (s.has_card_in_hand("禁じられた聖槍") or s.has_card_on_szone("禁じられた聖槍"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min - 800)
                )
            ),
        ),
        (
            "card_id:アレキサンドライドラゴン|command_type:SET_MONST|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and s.my_lp <= s.rival_monster_atk_sum
            and not (
                s.my_hand_max_normal_summon_atk >= s.rival_monster_atk_min
                or s.has_card_in_hand("ボマー・ドラゴン")
                or (
                    (s.has_card_in_hand("収縮") or s.has_card_on_szone("収縮"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min / 2)
                )
                or (
                    (s.has_card_in_hand("禁じられた聖槍") or s.has_card_on_szone("禁じられた聖槍"))
                    and s.my_hand_max_normal_summon_atk >= (s.rival_monster_atk_min - 800)
                )
            ),
        ),
        (
            "card_id:ボマー・ドラゴン|command_type:SUMMON|phase:NO_VALUE",
            lambda s: s.rival_monster_atk_min > s.my_hand_max_normal_summon_atk,
        ),
        ("card_id:アレキサンドライドラゴン|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        ("card_id:サファイアドラゴン|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        ("card_id:アサルトワイバーン|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        ("card_id:創世の竜騎士|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        ("card_id:マンジュ・ゴッド|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        ("card_id:センジュ・ゴッド|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        ("card_id:ソニックバード|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        ("card_id:仮面竜|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        ("card_id:ボマー・ドラゴン|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        (
            "card_id:死者蘇生|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and (
                (s.rival_monster_atk_max >= 3000)
                or (s.rival_monster_atk_max >= 1900 and s.my_lp <= 3000)
                or (
                    s.is_main2
                    and (s.my_lp <= 3000 or s.rival_monster_atk_sum >= s.my_lp)
                    and (s.has_card_in_grave("青眼の白龍") or s.has_rival_card_in_grave("青眼の白龍"))
                )
            ),
        ),
        (
            "card_id:銀龍の轟咆|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and (
                (s.rival_monster_atk_max >= 3000)
                or (s.rival_monster_atk_max >= 1900 and s.my_lp <= 3000)
                or (
                    s.is_main2
                    and (s.my_lp <= 3000 or s.rival_monster_atk_sum >= s.my_lp)
                    and s.has_card_in_grave("青眼の白龍")
                )
            ),
        ),
        (
            "card_id:早すぎた埋葬|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and ((s.rival_monster_atk_max >= 3000) or (s.rival_monster_atk_max >= 1900 and s.my_lp <= 3000)),
        ),
        (
            "card_id:リビングデッドの呼び声|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and (
                (s.rival_monster_atk_max >= 3000)
                or (s.rival_monster_atk_max >= 1900 and s.my_lp <= 3000)
                or (
                    s.is_main2
                    and (s.my_lp <= 3000 or s.rival_monster_atk_sum >= s.my_lp)
                    and s.has_card_in_grave("青眼の白龍")
                )
            ),
        ),
        (
            "card_id:強化蘇生|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.my_mzone_count == 0
            and (
                (s.rival_monster_atk_max >= 3000)
                or (s.rival_monster_atk_max >= 1900 and s.my_lp <= 3000)
                or (
                    s.is_main2
                    and (s.my_lp <= 3000 or s.rival_monster_atk_sum >= s.my_lp)
                    and s.has_card_in_grave("洞窟に潜む竜")
                )
            ),
        ),
        (
            "card_id:戦線復帰|command_type:ACTIVATE|phase:NO_VALUE",
            lambda s: s.is_main2
            and s.my_mzone_count == 0
            and (s.my_lp <= 3000 or s.rival_monster_atk_sum >= s.my_lp)
            and (s.has_card_in_grave("青眼の白龍") or s.has_card_in_grave("洞窟に潜む竜")),
        ),
        (
            "card_id:聖なるバリア －ミラーフォース－|command_type:SET|phase:NO_VALUE",
            lambda s: s.my_szone_count
            < (
                5
                if s.has_rival_card_in_grave("大嵐")
                else (
                    4
                    if s.my_lp <= 4000
                    else (
                        2
                        if s.is_first_turn
                        else (3 if s.my_hand_count >= 5 else (2 if s.rival_deck_count <= 18 else 3))
                    )
                )
            )
            and (s.rival_mzone_count >= 1 or s.my_mzone_count == 0),
        ),
        (
            "card_id:激流葬|command_type:SET|phase:NO_VALUE",
            lambda s: s.my_szone_count
            < (
                5
                if s.has_rival_card_in_grave("大嵐")
                else (
                    4
                    if s.my_lp <= 4000
                    else (
                        2
                        if s.is_first_turn
                        else (3 if s.my_hand_count >= 5 else (2 if s.rival_deck_count <= 18 else 3))
                    )
                )
            )
            and (s.rival_mzone_count >= 1 or s.my_mzone_count == 0),
        ),
        (
            "card_id:禁じられた聖槍|command_type:SET|phase:NO_VALUE",
            lambda s: s.my_szone_count
            < (
                5
                if s.has_rival_card_in_grave("大嵐")
                else (
                    4
                    if s.my_lp <= 4000
                    else (
                        2
                        if s.is_first_turn
                        else (3 if s.my_hand_count >= 5 else (2 if s.rival_deck_count <= 18 else 3))
                    )
                )
            )
            and (s.is_main2 or s.is_first_turn)
            and (s.rival_mzone_count >= 1 or (s.my_lp <= 5000 and s.my_mzone_count == 0)),
        ),
        (
            "card_id:月の書|command_type:SET|phase:NO_VALUE",
            lambda s: s.my_szone_count
            < (
                5
                if s.has_rival_card_in_grave("大嵐")
                else (
                    4
                    if s.my_lp <= 4000
                    else (
                        2
                        if s.is_first_turn
                        else (3 if s.my_hand_count >= 5 else (2 if s.rival_deck_count <= 18 else 3))
                    )
                )
            )
            and (s.is_main2 or s.is_first_turn)
            and (s.rival_mzone_count >= 1 or (s.my_lp <= 5000 and s.my_mzone_count == 0)),
        ),
        (
            "card_id:収縮|command_type:SET|phase:NO_VALUE",
            lambda s: s.my_szone_count
            < (
                5
                if s.has_rival_card_in_grave("大嵐")
                else (
                    4
                    if s.my_lp <= 4000
                    else (
                        2
                        if s.is_first_turn
                        else (3 if s.my_hand_count >= 5 else (2 if s.rival_deck_count <= 18 else 3))
                    )
                )
            )
            and (s.is_main2 or s.is_first_turn)
            and (s.rival_mzone_count >= 1 or (s.my_lp <= 5000 and s.my_mzone_count == 0)),
        ),
        (
            "card_id:銀龍の轟咆|command_type:SET|phase:NO_VALUE",
            lambda s: s.my_szone_count
            < (
                5
                if s.has_rival_card_in_grave("大嵐")
                else (
                    4
                    if s.my_lp <= 4000
                    else (
                        2
                        if s.is_first_turn
                        else (3 if s.my_hand_count >= 5 else (2 if s.rival_deck_count <= 18 else 3))
                    )
                )
            )
            and (s.is_main2 or s.is_first_turn)
            and (
                any(
                    s.has_card_in_grave(n) or s.has_card_on_mzone(n)
                    for n in ["青眼の白龍", "アレキサンドライドラゴン", "サファイアドラゴン", "洞窟に潜む竜"]
                )
            ),
        ),
        (
            "card_id:リビングデッドの呼び声|command_type:SET|phase:NO_VALUE",
            lambda s: s.my_szone_count
            < (
                5
                if s.has_rival_card_in_grave("大嵐")
                else (
                    4
                    if s.my_lp <= 4000
                    else (
                        2
                        if s.is_first_turn
                        else (3 if s.my_hand_count >= 5 else (2 if s.rival_deck_count <= 18 else 3))
                    )
                )
            )
            and (
                (s.has_card_in_grave("青眼の白龍") or s.has_card_on_mzone("青眼の白龍"))
                or (s.my_grave_monster_atk_max >= 1800 or s.my_monster_atk_max >= 1800)
                or (s.my_mzone_count == 0 and s.my_grave_monster_count >= 1)
            ),
        ),
        (
            "card_id:戦線復帰|command_type:SET|phase:NO_VALUE",
            lambda s: s.my_szone_count
            < (
                5
                if s.has_rival_card_in_grave("大嵐")
                else (
                    4
                    if s.my_lp <= 4000
                    else (
                        2
                        if s.is_first_turn
                        else (3 if s.my_hand_count >= 5 else (2 if s.rival_deck_count <= 18 else 3))
                    )
                )
            )
            and (
                (s.has_card_in_grave("青眼の白龍") or s.has_card_on_mzone("青眼の白龍"))
                or (s.my_grave_monster_atk_max >= 1800 or s.my_monster_atk_max >= 1800)
                or (s.my_mzone_count == 0 and s.my_grave_monster_count >= 1)
            ),
        ),
        (
            "card_id:強化蘇生|command_type:SET|phase:NO_VALUE",
            lambda s: s.my_szone_count
            < (
                5
                if s.has_rival_card_in_grave("大嵐")
                else (
                    4
                    if s.my_lp <= 4000
                    else (
                        2
                        if s.is_first_turn
                        else (3 if s.my_hand_count >= 5 else (2 if s.rival_deck_count <= 18 else 3))
                    )
                )
            )
            and (s.my_level4_or_lower_monster_grave_count >= 1 or s.my_level4_or_lower_monster_mzone_count >= 1),
        ),
        (
            "card_id:サイクロン|command_type:SET|phase:NO_VALUE",
            lambda s: s.my_szone_count
            < (
                5
                if s.has_rival_card_in_grave("大嵐")
                else (
                    4
                    if s.my_lp <= 4000
                    else (
                        2
                        if s.is_first_turn
                        else (3 if s.my_hand_count >= 5 else (2 if s.rival_deck_count <= 18 else 3))
                    )
                )
            )
            and (s.is_main2 or s.is_first_turn)
            and (s.rival_szone_count >= 1 or s.my_lp <= 5000 or s.is_first_turn),
        ),
        (
            "card_id:砂塵の大竜巻|command_type:SET|phase:NO_VALUE",
            lambda s: s.my_szone_count
            < (
                5
                if s.has_rival_card_in_grave("大嵐")
                else (
                    4
                    if s.my_lp <= 4000
                    else (
                        2
                        if s.is_first_turn
                        else (3 if s.my_hand_count >= 5 else (2 if s.rival_deck_count <= 18 else 3))
                    )
                )
            )
            and (s.rival_szone_count >= 1 or s.my_lp <= 5000 or s.is_first_turn),
        ),
        ("card_id:コドモドラゴン|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        (
            "card_id:洞窟に潜む竜|command_type:TURN_DEF|phase:NO_VALUE",
            lambda s: s.is_main2
            and (
                (
                    s.rival_mzone_count >= 1
                    and s.my_lp <= s.rival_monster_atk_sum
                    and CARD_MAP["洞窟に潜む竜"]["atk"] < s.rival_monster_atk_min
                )
                or (
                    s.rival_mzone_count >= 1
                    and CARD_MAP["洞窟に潜む竜"]["atk"] < s.rival_monster_atk_max
                    and CARD_MAP["洞窟に潜む竜"]["def"] > s.rival_monster_atk_max
                )
                or (s.has_rival_card_on_mzone("青眼の白龍") and s.my_mzone_count >= 2)
            ),
        ),
        (
            "card_id:サファイアドラゴン|command_type:TURN_DEF|phase:NO_VALUE",
            lambda s: s.is_main2
            and (
                s.rival_mzone_count >= 1
                and s.my_lp <= s.rival_monster_atk_sum
                and CARD_MAP["サファイアドラゴン"]["atk"] < s.rival_monster_atk_min
            ),
        ),
        (
            "card_id:青眼の白龍|command_type:TURN_DEF|phase:NO_VALUE",
            lambda s: s.is_main2
            and (
                s.rival_mzone_count >= 1
                and s.my_lp <= s.rival_monster_atk_sum
                and CARD_MAP["青眼の白龍"]["atk"] < s.rival_monster_atk_min
            ),
        ),
        ("card_id:None|command_type:CHANGE_PHASE|phase:BATTLE", lambda s: True),
        ("card_id:None|command_type:CHANGE_PHASE|phase:END", lambda s: True),
        ("card_id:ライトニング・ボルテックス|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:大嵐|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:サイクロン|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:砂塵の大竜巻|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:白竜降臨|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:白竜の聖騎士|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:創世の竜騎士|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:死者蘇生|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:銀龍の轟咆|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:早すぎた埋葬|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:リビングデッドの呼び声|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:戦線復帰|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:強化蘇生|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:月の書|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:収縮|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:禁じられた聖槍|command_type:ACTIVATE|phase:NO_VALUE", lambda s: True),
        ("card_id:青眼の白龍|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        ("card_id:洞窟に潜む竜|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        ("card_id:コドモドラゴン|command_type:SUMMON|phase:NO_VALUE", lambda s: True),
        ("card_id:マンジュ・ゴッド|command_type:REVERSE|phase:NO_VALUE", lambda s: True),
        ("card_id:青眼の白龍|command_type:REVERSE|phase:NO_VALUE", lambda s: True),
        ("card_id:青眼の白龍|command_type:TURN_ATK|phase:NO_VALUE", lambda s: True),
        ("card_id:白竜の聖騎士|command_type:TURN_ATK|phase:NO_VALUE", lambda s: True),
        ("card_id:アレキサンドライドラゴン|command_type:TURN_ATK|phase:NO_VALUE", lambda s: True),
        ("card_id:ボマー・ドラゴン|command_type:TURN_ATK|phase:NO_VALUE", lambda s: True),
        ("card_id:洞窟に潜む竜|command_type:TURN_ATK|phase:NO_VALUE", lambda s: True),
        ("card_id:コドモドラゴン|command_type:REVERSE|phase:NO_VALUE", lambda s: True),
        ("card_id:洞窟に潜む竜|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:創世の竜騎士|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:サファイアドラゴン|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:仮面竜|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:アサルトワイバーン|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:マンジュ・ゴッド|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:センジュ・ゴッド|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:ソニックバード|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:コドモドラゴン|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:アレキサンドライドラゴン|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:ボマー・ドラゴン|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:青眼の白龍|command_type:SET_MONST|phase:NO_VALUE", lambda s: True),
        ("card_id:聖なるバリア －ミラーフォース－|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:激流葬|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:禁じられた聖槍|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:月の書|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:収縮|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:銀龍の轟咆|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:リビングデッドの呼び声|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:戦線復帰|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:強化蘇生|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:サイクロン|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:砂塵の大竜巻|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:強欲な壺|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:大嵐|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:ライトニング・ボルテックス|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:早すぎた埋葬|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:死者蘇生|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:白竜降臨|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:高等儀式術|command_type:SET|phase:NO_VALUE", lambda s: True),
        ("card_id:洞窟に潜む竜|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:サファイアドラゴン|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:アレキサンドライドラゴン|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:青眼の白龍|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:アサルトワイバーン|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:仮面竜|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:ボマー・ドラゴン|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:コドモドラゴン|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:センジュ・ゴッド|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:ソニックバード|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:マンジュ・ゴッド|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:創世の竜騎士|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
        ("card_id:白竜の聖騎士|command_type:TURN_DEF|phase:NO_VALUE", lambda s: True),
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
        # `pos_id`選択
        magic_positions = [PosId.MAGIC_L_L, PosId.MAGIC_L, PosId.MAGIC_C, PosId.MAGIC_R, PosId.MAGIC_R_R]
        random.shuffle(magic_positions)

        for priority_pos in magic_positions + [PosId.HAND]:
            if filtered_indices := [i for i in candidate_indices if selectable_commands[i].pos_id == priority_pos]:
                candidate_indices = filtered_indices
                break

        command_type = selectable_commands[candidate_indices[0]].command_type

        def get_compare_stat(i: int, attr: str) -> int:
            card = state.duel_state_data.duel_card_table[selectable_commands[i].table_index]
            return getattr(card, attr)

        if command_type in [CommandType.REVERSE, CommandType.TURN_ATK]:
            max_stat = max(get_compare_stat(i, "atk_val") for i in candidate_indices)
            candidate_indices = [i for i in candidate_indices if get_compare_stat(i, "atk_val") == max_stat]

        if command_type == CommandType.TURN_DEF:
            max_stat = max(get_compare_stat(i, "def_val") for i in candidate_indices)
            candidate_indices = [i for i in candidate_indices if get_compare_stat(i, "def_val") == max_stat]

        if command_type == CommandType.ACTIVATE:
            min_stat = min(get_compare_stat(i, "atk_val") for i in candidate_indices)
            candidate_indices = [i for i in candidate_indices if get_compare_stat(i, "atk_val") == min_stat]

        target_pos_id = random.choice([selectable_commands[i].pos_id for i in candidate_indices])
        candidate_indices = [i for i in candidate_indices if selectable_commands[i].pos_id == target_pos_id]

        #  ランダム選択 (`card_index`, `table_index`)
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
