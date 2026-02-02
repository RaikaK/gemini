import random
from typing import Callable

from ygo.constants.enums import Face
from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.chain_effect_ritual_release import OPTIONS
from ..utils import Situation, write_debug_log


def select_chain_effect_ritual_release(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """チェーンの効果処理中 (SelectionType:10) & 儀式召喚に必要なレベル分のモンスターをリリースしてください。 (SelectionId:19)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        ("card_id:洞窟に潜む竜|pos_id:HAND", lambda s: True),
        ("card_id:アレキサンドライドラゴン|pos_id:HAND", lambda s: True),
        (
            "card_id:アサルトワイバーン|pos_id:HAND",
            lambda s: s.my_hand_count >= 5
            and not (s.has_card_in_grave("青眼の白龍") or s.has_card_in_hand("青眼の白龍")),
        ),
        (
            "card_id:ソニックバード|pos_id:HAND",
            lambda s: s.my_hand_count >= 5 or s.has_card_in_hand("白竜降臨") or s.has_card_in_hand("高等儀式術"),
        ),
        (
            "card_id:センジュ・ゴッド|pos_id:HAND",
            lambda s: s.my_hand_count >= 5 or s.has_card_in_hand("白竜の聖騎士"),
        ),
        ("card_id:ソニックバード|pos_id:MONSTER", lambda s: s.my_mzone_count >= 3),
        ("card_id:センジュ・ゴッド|pos_id:MONSTER", lambda s: s.my_mzone_count >= 3),
        ("card_id:マンジュ・ゴッド|pos_id:MONSTER", lambda s: s.my_mzone_count >= 3),
        ("card_id:ソニックバード|pos_id:HAND", lambda s: True),
        ("card_id:センジュ・ゴッド|pos_id:HAND", lambda s: True),
        ("card_id:ソニックバード|pos_id:MONSTER", lambda s: True),
        ("card_id:センジュ・ゴッド|pos_id:MONSTER", lambda s: True),
        ("card_id:マンジュ・ゴッド|pos_id:MONSTER", lambda s: True),
        ("card_id:洞窟に潜む竜|pos_id:MONSTER", lambda s: True),
        ("card_id:アサルトワイバーン|pos_id:HAND", lambda s: True),
        ("card_id:アサルトワイバーン|pos_id:MONSTER", lambda s: True),
        ("card_id:ボマー・ドラゴン|pos_id:MONSTER", lambda s: True),
        ("card_id:白竜の聖騎士|pos_id:HAND", lambda s: True),
        ("card_id:青眼の白龍|pos_id:HAND", lambda s: True),
        ("card_id:青眼の白龍|pos_id:MONSTER", lambda s: True),
    ]

    # 選択可能な行動を評価
    command_scores: list[float] = []

    for target_command in selectable_commands:
        target_command_score: float = float("-inf")
        target_command_identifier: str | None = None

        # 識別子を特定
        for command_identifier, command_condition in OPTIONS.items():
            if isinstance(command_condition, dict):
                if all(
                    (
                        getattr(target_command, attr, None) in val
                        if isinstance(val, list)
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
        # `pos_id`選択 (`atk_val` or `def_val` 最小値)
        def get_compare_stat(i: int) -> int:
            card = state.duel_state_data.duel_card_table[selectable_commands[i].table_index]
            return card.atk_val if card.turn == Face.FRONT else card.def_val

        min_stat = min(get_compare_stat(i) for i in candidate_indices)
        candidate_indices = [i for i in candidate_indices if get_compare_stat(i) == min_stat]
        target_pos_id = random.choice([selectable_commands[i].pos_id for i in candidate_indices])
        candidate_indices = [i for i in candidate_indices if selectable_commands[i].pos_id == target_pos_id]

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
