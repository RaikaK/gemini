import random
from typing import Callable

from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.chain_before_destroy import OPTIONS
from ..utils import CARD_MAP, Situation, write_debug_log


def select_chain_before_destroy(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """チェーンに積まれる前の処理中 (SelectionType:9) & 破壊するカードを選択してください。 (SelectionId:15)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        (
            "card_id:早すぎた埋葬|player_id:RIVAL",
            lambda s: any(
                s.is_rival_monster_equipped(name)
                for name in ["青眼の白龍", "白竜の聖騎士", "アレキサンドライドラゴン", "サファイアドラゴン"]
            ),
        ),
        (
            "card_id:早すぎた埋葬|player_id:RIVAL",
            lambda s: s.has_rival_card_in_chain_stack("早すぎた埋葬"),
        ),
        (
            "card_id:リビングデッドの呼び声|player_id:RIVAL",
            lambda s: s.has_rival_card_in_chain_stack("リビングデッドの呼び声"),
        ),
        (
            "card_id:強化蘇生|player_id:RIVAL",
            lambda s: s.has_rival_card_in_chain_stack("強化蘇生"),
        ),
        (
            "card_id:強化蘇生|player_id:RIVAL",
            lambda s: any(
                s.is_enhanced_rival_monster_exists(name) for name, data in CARD_MAP.items() if data["monster"]
            )
            and s.my_monster_atk_max < s.rival_monster_atk_max
            and (s.my_monster_atk_max >= s.rival_monster_atk_max - 100),
        ),
        (
            "card_id:収縮|player_id:MYSELF",
            lambda s: s.my_szone_count >= 5
            and (s.has_card_in_hand("聖なるバリア －ミラーフォース－") or s.has_card_in_hand("激流葬")),
        ),
        (
            "card_id:強化蘇生|player_id:MYSELF",
            lambda s: s.my_szone_count >= 5
            and (s.has_card_in_hand("聖なるバリア －ミラーフォース－") or s.has_card_in_hand("激流葬")),
        ),
        (
            "card_id:銀龍の轟咆|player_id:MYSELF",
            lambda s: s.my_szone_count >= 5
            and (s.has_card_in_hand("聖なるバリア －ミラーフォース－") or s.has_card_in_hand("激流葬")),
        ),
        ("card_id:リビングデッドの呼び声|player_id:RIVAL", lambda s: True),
        ("card_id:Unknown|player_id:RIVAL", lambda s: True),
        ("card_id:早すぎた埋葬|player_id:RIVAL", lambda s: True),
        ("card_id:強化蘇生|player_id:RIVAL", lambda s: True),
        ("card_id:大嵐|player_id:RIVAL", lambda s: True),
        ("card_id:月の書|player_id:RIVAL", lambda s: True),
        ("card_id:収縮|player_id:MYSELF", lambda s: True),
        ("card_id:強化蘇生|player_id:MYSELF", lambda s: True),
        ("card_id:銀龍の轟咆|player_id:MYSELF", lambda s: True),
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
        # ランダム選択 (`pos_id`, `table_index`)
        def command_signature(i: int):
            return {
                key: val for key, val in vars(selectable_commands[i]).items() if key not in ("pos_id", "table_index")
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
