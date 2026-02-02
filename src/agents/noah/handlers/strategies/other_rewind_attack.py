from typing import Callable

from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.other_rewind_attack import OPTIONS
from ..utils import CARD_MAP, Situation, write_debug_log


def select_other_rewind_attack(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """その他 (SelectionType:12) & 戦闘が巻き戻されました。攻撃を続けますか？ (SelectionId:20)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        (
            "yes_no:YES",
            lambda s: s.my_attacking_monster_id == CARD_MAP["ボマー・ドラゴン"]["id"],
        ),
        (
            "yes_no:YES",
            lambda s: s.has_rival_face_card_on_mzone("ボマー・ドラゴン")
            and s.my_attacking_monster_id == CARD_MAP["仮面竜"]["id"],
        ),
        (
            "yes_no:YES",
            lambda s: s.has_rival_face_card_on_mzone("ボマー・ドラゴン") and s.my_attacking_monster_atk < 1900,
        ),
        (
            "yes_no:NO",
            lambda s: s.has_rival_face_card_on_mzone("ボマー・ドラゴン"),
        ),
        (
            "yes_no:YES",
            lambda s: s.rival_face_up_atk_monster_count == 1 and s.my_attacking_monster_atk > s.rival_monster_atk_max,
        ),
        (
            "yes_no:YES",
            lambda s: s.rival_face_up_def_monster_count == 1 and s.my_attacking_monster_atk > s.rival_monster_def_max,
        ),
        (
            "yes_no:YES",
            lambda s: s.rival_face_down_monster_count == 1
            and s.my_attacking_monster_atk > CARD_MAP["洞窟に潜む竜"]["def"],
        ),
        (
            "yes_no:NO",
            lambda s: s.rival_face_up_atk_monster_count == 1 and s.rival_monster_atk_max >= s.my_attacking_monster_atk,
        ),
        (
            "yes_no:NO",
            lambda s: s.rival_face_up_def_monster_count == 1 and s.rival_monster_def_max >= s.my_attacking_monster_atk,
        ),
        (
            "yes_no:NO",
            lambda s: s.rival_face_down_monster_count == 1
            and s.my_attacking_monster_atk <= CARD_MAP["洞窟に潜む竜"]["def"],
        ),
        ("yes_no:NO", lambda s: True),
        ("yes_no:YES", lambda s: True),
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
        # 同点は存在しない
        write_debug_log(
            selectable_commands,
            selection_type,
            selection_id,
            f"Ambiguous commands: {[selectable_commands[i] for i in candidate_indices]}",
        )

    return best_command_index
