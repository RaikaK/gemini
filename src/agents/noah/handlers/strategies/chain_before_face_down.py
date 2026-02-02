import random
from typing import Callable

from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.chain_before_face_down import OPTIONS
from ..utils import CARD_MAP, Situation, write_debug_log


def select_chain_before_face_down(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """チェーンに積まれる前の処理中 (SelectionType:9) & 裏側守備表示にするモンスターを選択してください。 (SelectionId:27)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        (
            "card_id:青眼の白龍|player_id:MYSELF",
            lambda s: s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            or s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－"),
        ),
        (
            "card_id:青眼の白龍|player_id:MYSELF",
            lambda s: s.is_my_card_targeted
            and (s.has_rival_card_in_chain_stack("収縮") or s.has_rival_card_in_chain_stack("禁じられた聖槍")),
        ),
        (
            "card_id:アレキサンドライドラゴン|player_id:MYSELF",
            lambda s: s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            or s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－"),
        ),
        (
            "card_id:仮面竜|player_id:MYSELF",
            lambda s: s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            or s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－"),
        ),
        (
            "card_id:マンジュ・ゴッド|player_id:MYSELF",
            lambda s: s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            or s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－"),
        ),
        (
            "card_id:センジュ・ゴッド|player_id:MYSELF",
            lambda s: s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            or s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－"),
        ),
        (
            "card_id:ソニックバード|player_id:MYSELF",
            lambda s: s.has_rival_card_in_chain_stack("ライトニング・ボルテックス")
            or s.has_rival_card_in_chain_stack("聖なるバリア －ミラーフォース－"),
        ),
        (
            "card_id:青眼の白龍|player_id:RIVAL",
            lambda s: s.is_my_turn
            and (s.is_main1 or s.is_battle_phase)
            and s.my_monster_atk_max < CARD_MAP["青眼の白龍"]["atk"]
            and s.my_monster_atk_max >= CARD_MAP["青眼の白龍"]["def"],
        ),
        (
            "card_id:アレキサンドライドラゴン|player_id:RIVAL",
            lambda s: s.is_my_turn
            and (s.is_main1 or s.is_battle_phase)
            and s.my_monster_atk_max < CARD_MAP["アレキサンドライドラゴン"]["atk"]
            and s.my_monster_atk_max >= CARD_MAP["アレキサンドライドラゴン"]["def"],
        ),
        (
            "card_id:サファイアドラゴン|player_id:RIVAL",
            lambda s: s.is_my_turn
            and (s.is_main1 or s.is_battle_phase)
            and s.my_monster_atk_max < CARD_MAP["サファイアドラゴン"]["atk"]
            and s.my_monster_atk_max >= CARD_MAP["サファイアドラゴン"]["def"],
        ),
        (
            "card_id:アサルトワイバーン|player_id:RIVAL",
            lambda s: s.is_my_turn
            and (s.is_main1 or s.is_battle_phase)
            and s.my_monster_atk_max < CARD_MAP["アサルトワイバーン"]["atk"]
            and s.my_monster_atk_max >= CARD_MAP["アサルトワイバーン"]["def"],
        ),
        (
            "card_id:創世の竜騎士|player_id:RIVAL",
            lambda s: s.is_my_turn
            and (s.is_main1 or s.is_battle_phase)
            and s.my_monster_atk_max < CARD_MAP["創世の竜騎士"]["atk"]
            and s.my_monster_atk_max >= CARD_MAP["創世の竜騎士"]["def"],
        ),
        (
            "card_id:仮面竜|player_id:RIVAL",
            lambda s: s.is_my_turn
            and (s.is_main1 or s.is_battle_phase)
            and s.my_monster_atk_max < CARD_MAP["仮面竜"]["atk"]
            and s.my_monster_atk_max >= CARD_MAP["仮面竜"]["def"],
        ),
        (
            "card_id:マンジュ・ゴッド|player_id:RIVAL",
            lambda s: s.is_my_turn
            and (s.is_main1 or s.is_battle_phase)
            and s.my_monster_atk_max < CARD_MAP["マンジュ・ゴッド"]["atk"]
            and s.my_monster_atk_max >= CARD_MAP["マンジュ・ゴッド"]["def"],
        ),
        (
            "card_id:ボマー・ドラゴン|player_id:RIVAL",
            lambda s: s.is_my_turn
            and (s.is_main1 or s.is_battle_phase)
            and s.my_monster_atk_max < CARD_MAP["ボマー・ドラゴン"]["atk"]
            and s.my_monster_atk_max >= CARD_MAP["ボマー・ドラゴン"]["def"],
        ),
        (
            "card_id:青眼の白龍|player_id:RIVAL",
            lambda s: s.is_rival_turn and s.rival_attacking_monster_id == CARD_MAP["青眼の白龍"]["id"],
        ),
        (
            "card_id:アサルトワイバーン|player_id:RIVAL",
            lambda s: s.is_rival_turn and s.rival_attacking_monster_id == CARD_MAP["アサルトワイバーン"]["id"],
        ),
        (
            "card_id:アレキサンドライドラゴン|player_id:RIVAL",
            lambda s: s.is_rival_turn and s.rival_attacking_monster_id == CARD_MAP["アレキサンドライドラゴン"]["id"],
        ),
        (
            "card_id:サファイアドラゴン|player_id:RIVAL",
            lambda s: s.is_rival_turn and s.rival_attacking_monster_id == CARD_MAP["サファイアドラゴン"]["id"],
        ),
        (
            "card_id:ボマー・ドラゴン|player_id:RIVAL",
            lambda s: s.is_rival_turn and s.rival_attacking_monster_id == CARD_MAP["ボマー・ドラゴン"]["id"],
        ),
        (
            "card_id:創世の竜騎士|player_id:RIVAL",
            lambda s: s.is_rival_turn and s.rival_attacking_monster_id == CARD_MAP["創世の竜騎士"]["id"],
        ),
        (
            "card_id:マンジュ・ゴッド|player_id:MYSELF",
            lambda s: s.has_face_card_on_mzone("マンジュ・ゴッド")
            and s.is_main1
            and (
                not s.has_card_in_hand("白竜の聖騎士")
                or (not s.has_card_in_hand("白竜降臨") and not s.has_card_in_hand("高等儀式術"))
            )
            and s.my_lp > 4000,
        ),
        (
            "card_id:センジュ・ゴッド|player_id:MYSELF",
            lambda s: s.has_face_card_on_mzone("センジュ・ゴッド")
            and not s.has_card_in_hand("白竜の聖騎士")
            and s.is_main1
            and s.my_lp > 4000,
        ),
        (
            "card_id:ソニックバード|player_id:MYSELF",
            lambda s: s.has_face_card_on_mzone("ソニックバード")
            and (not s.has_card_in_hand("白竜降臨") and not s.has_card_in_hand("高等儀式術"))
            and s.is_main1
            and s.my_lp > 4000,
        ),
        ("card_id:青眼の白龍|player_id:RIVAL", lambda s: True),
        ("card_id:アレキサンドライドラゴン|player_id:RIVAL", lambda s: True),
        ("card_id:創世の竜騎士|player_id:RIVAL", lambda s: True),
        ("card_id:アサルトワイバーン|player_id:RIVAL", lambda s: True),
        ("card_id:サファイアドラゴン|player_id:RIVAL", lambda s: True),
        ("card_id:マンジュ・ゴッド|player_id:RIVAL", lambda s: True),
        ("card_id:仮面竜|player_id:RIVAL", lambda s: True),
        ("card_id:ボマー・ドラゴン|player_id:RIVAL", lambda s: True),
        ("card_id:青眼の白龍|player_id:MYSELF", lambda s: True),
        ("card_id:仮面竜|player_id:MYSELF", lambda s: True),
        ("card_id:ソニックバード|player_id:MYSELF", lambda s: True),
        ("card_id:センジュ・ゴッド|player_id:MYSELF", lambda s: True),
        ("card_id:マンジュ・ゴッド|player_id:MYSELF", lambda s: True),
        ("card_id:アレキサンドライドラゴン|player_id:MYSELF", lambda s: True),
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
