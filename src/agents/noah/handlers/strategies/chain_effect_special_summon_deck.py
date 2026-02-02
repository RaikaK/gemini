import random
from typing import Callable

from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.chain_effect_special_summon_deck import OPTIONS
from ..utils import CARD_MAP, Situation, write_debug_log


def select_chain_effect_special_summon_deck(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """チェーンの効果処理中 (SelectionType:10) & デッキから特殊召喚するモンスターを選択してください。 (SelectionId:3)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        (
            "card_id:ボマー・ドラゴン",
            lambda s: s.rival_monster_atk_max >= CARD_MAP["青眼の白龍"]["atk"]
            and s.my_monster_atk_max < s.rival_monster_atk_max,
        ),
        (
            "card_id:コドモドラゴン",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and s.has_card_in_hand("青眼の白龍")
            and s.rival_mzone_count >= 2,
        ),
        (
            "card_id:ボマー・ドラゴン",
            lambda s: s.rival_monster_atk_max >= 2000 and s.my_monster_atk_max < s.rival_monster_atk_max,
        ),
        (
            "card_id:仮面竜",
            lambda s: s.is_rival_turn and s.rival_mzone_count >= 2 and s.my_lp > 4000 and s.my_deck_count > 20,
        ),
        (
            "card_id:ボマー・ドラゴン",
            lambda s: s.has_rival_card_on_mzone("ボマー・ドラゴン"),
        ),
        (
            "card_id:洞窟に潜む竜",
            lambda s: 0 < s.rival_monster_atk_max <= 1900 and s.is_rival_turn,
        ),
        (
            "card_id:仮面竜",
            lambda s: s.my_deck_count > 20,
        ),
        ("card_id:ボマー・ドラゴン", lambda s: s.rival_mzone_count > 0 and s.my_mzone_count == 0),
        (
            "card_id:仮面竜",
            lambda s: s.my_lp <= 4000 or s.my_deck_count <= 20,
        ),
        ("card_id:コドモドラゴン", lambda s: s.has_card_in_hand("青眼の白龍")),
        ("card_id:仮面竜", lambda s: True),
        ("card_id:ボマー・ドラゴン", lambda s: True),
        ("card_id:洞窟に潜む竜", lambda s: True),
        ("card_id:コドモドラゴン", lambda s: True),
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
