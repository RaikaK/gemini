# 仮面竜->ボマー・ドラゴンのコンボ発動条件を考える
import random
from typing import Callable

from ygo.constants.enums import CommandType, PosId
from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.battle_phase import OPTIONS
from ..utils import CARD_MAP, Situation, write_debug_log


def select_battle_phase(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """バトルフェイズ (SelectionType:2) & 値無し (SelectionId:-1)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        (
            "card_id:白竜の聖騎士|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE",
            lambda s: s.rival_face_down_monster_count > 0,
        ),
        (
            "card_id:白竜の聖騎士|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE",
            lambda s: s.rival_face_down_monster_count > 0,
        ),
        (
            "card_id:コドモドラゴン|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE",
            lambda s: s.rival_mzone_count == 0,
        ),
        (
            "card_id:コドモドラゴン|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE",
            lambda s: s.rival_mzone_count == 0,
        ),
        (
            "card_id:洞窟に潜む竜|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["洞窟に潜む竜"]["atk"]
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, CARD_MAP["洞窟に潜む竜"]["atk"] - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or CARD_MAP["洞窟に潜む竜"]["atk"] - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:洞窟に潜む竜|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["洞窟に潜む竜"]["atk"] + 100
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, (CARD_MAP["洞窟に潜む竜"]["atk"] + 100) - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or (CARD_MAP["洞窟に潜む竜"]["atk"] + 100) - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:センジュ・ゴッド|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["センジュ・ゴッド"]["atk"]
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, CARD_MAP["センジュ・ゴッド"]["atk"] - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or CARD_MAP["センジュ・ゴッド"]["atk"] - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:センジュ・ゴッド|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["センジュ・ゴッド"]["atk"] + 100
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, (CARD_MAP["センジュ・ゴッド"]["atk"] + 100) - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or (CARD_MAP["センジュ・ゴッド"]["atk"] + 100) - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:ソニックバード|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["ソニックバード"]["atk"]
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, CARD_MAP["ソニックバード"]["atk"] - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or CARD_MAP["ソニックバード"]["atk"] - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:ソニックバード|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["ソニックバード"]["atk"] + 100
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, (CARD_MAP["ソニックバード"]["atk"] + 100) - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or (CARD_MAP["ソニックバード"]["atk"] + 100) - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:マンジュ・ゴッド|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["マンジュ・ゴッド"]["atk"]
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, CARD_MAP["マンジュ_ゴッド"]["atk"] - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or CARD_MAP["マンジュ・ゴッド"]["atk"] - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:マンジュ・ゴッド|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["マンジュ・ゴッド"]["atk"] + 100
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, (CARD_MAP["マンジュ・ゴッド"]["atk"] + 100) - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or (CARD_MAP["マンジュ・ゴッド"]["atk"] + 100) - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:仮面竜|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["仮面竜"]["atk"]
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, CARD_MAP["仮面竜"]["atk"] - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) == 1 and s.rival_hand_count <= 3
                    )
                    or CARD_MAP["仮面竜"]["atk"] - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:仮面竜|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["仮面竜"]["atk"] + 100
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, (CARD_MAP["仮面竜"]["atk"] + 100) - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) == 1 and s.rival_hand_count <= 3
                    )
                    or (CARD_MAP["仮面竜"]["atk"] + 100) - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:創世の竜騎士|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["創世の竜騎士"]["atk"]
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, CARD_MAP["創世の竜騎士"]["atk"] - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or CARD_MAP["創世の竜騎士"]["atk"] - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:創世の竜騎士|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["創世の竜騎士"]["atk"] + 100
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, (CARD_MAP["創世の竜騎士"]["atk"] + 100) - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or (CARD_MAP["創世の竜騎士"]["atk"] + 100) - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:サファイアドラゴン|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["サファイアドラゴン"]["atk"]
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, CARD_MAP["サファイアドラゴン"]["atk"] - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or CARD_MAP["サファイアドラゴン"]["atk"] - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:サファイアドラゴン|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["サファイアドラゴン"]["atk"] + 100
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, (CARD_MAP["サファイアドラゴン"]["atk"] + 100) - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or (CARD_MAP["サファイアドラゴン"]["atk"] + 100) - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:白竜の聖騎士|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["白竜の聖騎士"]["atk"]
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, CARD_MAP["白竜の聖騎士"]["atk"] - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) == 1 and s.rival_hand_count <= 3
                    )
                    or CARD_MAP["白竜の聖騎士"]["atk"] - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:白竜の聖騎士|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["白竜の聖騎士"]["atk"] + 100
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, (CARD_MAP["白竜の聖騎士"]["atk"] + 100) - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) == 1 and s.rival_hand_count <= 3
                    )
                    or (CARD_MAP["白竜の聖騎士"]["atk"] + 100) - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:アレキサンドライドラゴン|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["アレキサンドライドラゴン"]["atk"]
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, CARD_MAP["アレキサンドライドラゴン"]["atk"] - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or CARD_MAP["アレキサンドライドラゴン"]["atk"] - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        (
            "card_id:アレキサンドライドラゴン|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE",
            lambda s: s.rival_mzone_count == 0
            or s.rival_face_up_atk_monster_count_within(
                CARD_MAP["コドモドラゴン"]["atk"] + 101, CARD_MAP["アレキサンドライドラゴン"]["atk"] + 100
            )
            > 0
            or s.rival_face_up_def_monster_count_within(
                CARD_MAP["コドモドラゴン"]["def"] + 101, (CARD_MAP["アレキサンドライドラゴン"]["atk"] + 100) - 1
            )
            > 0
            or (
                (
                    s.rival_face_up_atk_monster_count_within(0, CARD_MAP["コドモドラゴン"]["atk"] + 100) > 0
                    or s.rival_face_up_def_monster_count_within(0, CARD_MAP["コドモドラゴン"]["def"] + 100) > 0
                )
                and (
                    s.rival_hand_count == 0
                    or s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
                    or (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 2
                    or (
                        (s.rival_blue_eyes_mzone_count + s.rival_blue_eyes_grave_count) >= 1 and s.rival_hand_count <= 2
                    )
                    or (CARD_MAP["アレキサンドライドラゴン"]["atk"] + 100) - s.rival_monster_atk_min >= s.rival_lp
                )
            ),
        ),
        ("card_id:None|command_type:CHANGE_PHASE|phase:MAIN2|is_powerful:FALSE", lambda s: True),
        ("card_id:None|command_type:CHANGE_PHASE|phase:END|is_powerful:FALSE", lambda s: True),
        ("card_id:コドモドラゴン|command_type:ATTACK|phase:NO_VALUE|is_powerful:FALSE", lambda s: True),
        ("card_id:コドモドラゴン|command_type:ATTACK|phase:NO_VALUE|is_powerful:TRUE", lambda s: True),
        ("card_id:洞窟に潜む竜|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:FALSE", lambda s: True),
        ("card_id:洞窟に潜む竜|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:TRUE", lambda s: True),
        ("card_id:センジュ・ゴッド|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:FALSE", lambda s: True),
        ("card_id:センジュ・ゴッド|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:TRUE", lambda s: True),
        ("card_id:ソニックバード|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:FALSE", lambda s: True),
        ("card_id:ソニックバード|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:TRUE", lambda s: True),
        ("card_id:マンジュ・ゴッド|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:FALSE", lambda s: True),
        ("card_id:マンジュ・ゴッド|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:TRUE", lambda s: True),
        ("card_id:創世の竜騎士|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:FALSE", lambda s: True),
        ("card_id:創世の竜騎士|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:TRUE", lambda s: True),
        ("card_id:サファイアドラゴン|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:FALSE", lambda s: True),
        ("card_id:サファイアドラゴン|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:TRUE", lambda s: True),
        ("card_id:アレキサンドライドラゴン|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:FALSE", lambda s: True),
        ("card_id:アレキサンドライドラゴン|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:TRUE", lambda s: True),
        ("card_id:収縮|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:FALSE", lambda s: True),
        ("card_id:月の書|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:FALSE", lambda s: True),
        ("card_id:禁じられた聖槍|command_type:ACTIVATE|phase:NO_VALUE|is_powerful:FALSE", lambda s: True),
    ]

    # 選択可能な行動を評価
    command_scores: list[float] = []

    for target_command in selectable_commands:
        target_command_score: float = float("-inf")
        target_command_identifier: str | None = None

        # 識別子を特定
        for command_identifier, command_condition in OPTIONS.items():
            if all(
                (
                    (
                        target_command.command_type == CommandType.ATTACK
                        and state.duel_state_data.duel_card_table[target_command.table_index].atk_val
                        > next(card["atk"] for card in CARD_MAP.values() if card["id"] == target_command.card_id)
                    )
                    == val
                    if attr == "is_powerful"
                    else getattr(target_command, attr, None) == val
                )
                for attr, val in command_condition.items()
            ):
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
