import random
from typing import Callable

from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.chain_before_monster import OPTIONS
from ..utils import CARD_MAP, Situation, write_debug_log


def select_chain_before_monster(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """チェーンに積まれる前の処理中 (SelectionType:9) & モンスターを選択してください。 (SelectionId:14)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        (
            "card_id:青眼の白龍|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and (
                s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－")
                or s.has_card_in_chain_stack("激流葬")
                or s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            )
            and s.has_face_card_on_mzone("青眼の白龍"),
        ),
        (
            "card_id:白竜の聖騎士|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and (
                s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－")
                or s.has_card_in_chain_stack("激流葬")
                or s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            )
            and s.has_face_card_on_mzone("白竜の聖騎士"),
        ),
        (
            "card_id:青眼の白龍|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and s.is_my_card_targeted
            and s.my_lp <= 2500
            and s.my_mzone_count == 1,
        ),
        (
            "card_id:アレキサンドライドラゴン|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and (
                s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－")
                or s.has_card_in_chain_stack("激流葬")
                or s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            )
            and s.has_face_card_on_mzone("アレキサンドライドラゴン"),
        ),
        (
            "card_id:アサルトワイバーン|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and (
                s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－")
                or s.has_card_in_chain_stack("激流葬")
                or s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            )
            and s.has_face_card_on_mzone("アサルトワイバーン"),
        ),
        (
            "card_id:青眼の白龍|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and s.is_my_card_targeted
            and s.has_face_card_on_mzone("青眼の白龍"),
        ),
        (
            "card_id:白竜の聖騎士|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and s.is_my_card_targeted
            and s.has_face_card_on_mzone("白竜の聖騎士"),
        ),
        (
            "card_id:サファイアドラゴン|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and (
                s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－")
                or s.has_card_in_chain_stack("激流葬")
                or s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            ),
        ),
        (
            "card_id:アサルトワイバーン|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and s.is_my_card_targeted
            and s.has_face_card_on_mzone("アサルトワイバーン"),
        ),
        (
            "card_id:マンジュ・ゴッド|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and (
                s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－")
                or s.has_card_in_chain_stack("激流葬")
                or s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            ),
        ),
        (
            "card_id:サファイアドラゴン|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"] and s.is_my_card_targeted,
        ),
        (
            "card_id:マンジュ・ゴッド|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"] and s.is_my_card_targeted,
        ),
        (
            "card_id:青眼の白龍|player_id:MYSELF",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and s.is_my_card_targeted
            and not s.is_my_monster_equipped("青眼の白龍"),
        ),
        (
            "card_id:青眼の白龍|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["収縮"]["id"]
            and s.is_dmg_before_calc
            and (s.is_my_monster_attacking or s.is_rival_monster_attacking)
            and s.rival_monster_atk_max >= 3000
            and (
                (s.is_rival_turn and (3000 // 2) <= s.my_monster_atk_max)
                or (s.is_my_turn and s.my_monster_atk_max >= (3000 // 2))
            ),
        ),
        (
            "card_id:青眼の白龍|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and s.is_dmg_before_calc
            and s.my_attacking_monster_atk > (CARD_MAP["青眼の白龍"]["atk"] - 800),
        ),
        (
            "card_id:アレキサンドライドラゴン|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and s.is_dmg_before_calc
            and s.my_attacking_monster_id == CARD_MAP["アサルトワイバーン"]["id"]
            and s.my_attacking_monster_atk > (CARD_MAP["アレキサンドライドラゴン"]["atk"] - 800),
        ),
        (
            "card_id:サファイアドラゴン|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["収縮"]["id"]
            and s.is_dmg_before_calc
            and (s.is_my_monster_attacking or s.is_rival_monster_attacking)
            and s.rival_monster_atk_max >= 1900
            and (
                (s.is_rival_turn and (1900 // 2) <= s.my_monster_atk_max)
                or (s.is_my_turn and s.my_monster_atk_max >= (1900 // 2))
            ),
        ),
        (
            "card_id:サファイアドラゴン|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and s.is_dmg_before_calc
            and s.my_attacking_monster_id == CARD_MAP["アサルトワイバーン"]["id"]
            and s.my_attacking_monster_atk > (CARD_MAP["サファイアドラゴン"]["atk"] - 800),
        ),
        (
            "card_id:アレキサンドライドラゴン|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["収縮"]["id"]
            and s.is_dmg_before_calc
            and (s.is_my_monster_attacking or s.is_rival_monster_attacking)
            and s.rival_monster_atk_max >= 2000
            and (
                (s.is_rival_turn and (2000 // 2) <= s.my_monster_atk_max)
                or (s.is_my_turn and s.my_monster_atk_max >= (2000 // 2))
            ),
        ),
        (
            "card_id:白竜の聖騎士|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["収縮"]["id"]
            and s.is_dmg_before_calc
            and (s.is_my_monster_attacking or s.is_rival_monster_attacking)
            and s.rival_monster_atk_max >= 1900
            and (
                (s.is_rival_turn and (1900 // 2) <= s.my_monster_atk_max)
                or (s.is_my_turn and s.my_monster_atk_max >= (1900 // 2))
            ),
        ),
        (
            "card_id:アレキサンドライドラゴン|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and s.is_dmg_before_calc
            and s.my_attacking_monster_atk > (CARD_MAP["アレキサンドライドラゴン"]["atk"] - 800)
            and s.has_rival_face_card_on_mzone("アレキサンドライドラゴン"),
        ),
        (
            "card_id:サファイアドラゴン|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["禁じられた聖槍"]["id"]
            and s.is_dmg_before_calc
            and s.my_attacking_monster_atk > (CARD_MAP["サファイアドラゴン"]["atk"] - 800)
            and s.has_rival_face_card_on_mzone("サファイアドラゴン"),
        ),
        (
            "card_id:ボマー・ドラゴン|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["収縮"]["id"]
            and s.has_rival_face_card_on_mzone("ボマー・ドラゴン"),
        ),
        (
            "card_id:仮面竜|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["収縮"]["id"]
            and s.is_battle_phase
            and s.has_rival_face_card_on_mzone("仮面竜"),
        ),
        (
            "card_id:マンジュ・ゴッド|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["収縮"]["id"]
            and s.has_rival_face_card_on_mzone("マンジュ・ゴッド"),
        ),
        (
            "card_id:ソニックバード|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["収縮"]["id"]
            and s.has_rival_face_card_on_mzone("ソニックバード"),
        ),
        (
            "card_id:センジュ・ゴッド|player_id:RIVAL",
            lambda s: s.last_action_card_id == CARD_MAP["収縮"]["id"]
            and s.has_rival_face_card_on_mzone("センジュ・ゴッド"),
        ),
        ("card_id:青眼の白龍|player_id:RIVAL", lambda s: True),
        ("card_id:アレキサンドライドラゴン|player_id:RIVAL", lambda s: True),
        ("card_id:白竜の聖騎士|player_id:RIVAL", lambda s: True),
        ("card_id:サファイアドラゴン|player_id:RIVAL", lambda s: True),
        ("card_id:アサルトワイバーン|player_id:RIVAL", lambda s: True),
        ("card_id:仮面竜|player_id:RIVAL", lambda s: True),
        ("card_id:ボマー・ドラゴン|player_id:RIVAL", lambda s: True),
        ("card_id:マンジュ・ゴッド|player_id:RIVAL", lambda s: True),
        ("card_id:ソニックバード|player_id:RIVAL", lambda s: True),
        ("card_id:センジュ・ゴッド|player_id:RIVAL", lambda s: True),
        ("card_id:青眼の白龍|player_id:MYSELF", lambda s: True),
        ("card_id:白竜の聖騎士|player_id:MYSELF", lambda s: True),
        ("card_id:アレキサンドライドラゴン|player_id:MYSELF", lambda s: True),
        ("card_id:サファイアドラゴン|player_id:MYSELF", lambda s: True),
        ("card_id:アサルトワイバーン|player_id:MYSELF", lambda s: True),
        ("card_id:マンジュ・ゴッド|player_id:MYSELF", lambda s: True),
        ("card_id:ソニックバード|player_id:MYSELF", lambda s: True),
        ("card_id:仮面竜|player_id:MYSELF", lambda s: True),
        ("card_id:洞窟に潜む竜|player_id:MYSELF", lambda s: True),
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
        # `pos_id`選択 (`atk_val` 最大値)
        def get_compare_stat(i: int) -> int:
            card = state.duel_state_data.duel_card_table[selectable_commands[i].table_index]
            return card.atk_val

        max_stat = max(get_compare_stat(i) for i in candidate_indices)
        candidate_indices = [i for i in candidate_indices if get_compare_stat(i) == max_stat]
        target_pos_id = random.choice([selectable_commands[i].pos_id for i in candidate_indices])
        candidate_indices = [i for i in candidate_indices if selectable_commands[i].pos_id == target_pos_id]

        # ランダム選択 (`table_index`)
        def command_signature(i: int):
            return {key: val for key, val in vars(selectable_commands[i]).items() if key not in ("table_index")}

        if any(command_signature(i) != command_signature(candidate_indices[0]) for i in candidate_indices):
            write_debug_log(
                selectable_commands,
                selection_type,
                selection_id,
                f"Ambiguous commands: {[selectable_commands[i] for i in candidate_indices]}",
            )

        best_command_index = random.choice(candidate_indices)

    return best_command_index
