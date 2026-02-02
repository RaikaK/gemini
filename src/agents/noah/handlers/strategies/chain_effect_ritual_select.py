import random
from typing import Callable

from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.chain_effect_ritual_select import OPTIONS
from ..utils import Situation, write_debug_log


def select_chain_effect_ritual_select(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """チェーンの効果処理中 (SelectionType:10) & 儀式召喚に必要なレベル分のモンスターを選択してください。 (SelectionId:18)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        (
            "card_id:アレキサンドライドラゴン",
            lambda s: s.has_card_in_hand("銀龍の轟咆")
            or s.has_card_on_szone("銀龍の轟咆")
            or s.has_card_in_hand("死者蘇生")
            or s.has_card_on_szone("死者蘇生")
            or s.has_card_in_hand("リビングデッドの呼び声")
            or s.has_card_on_szone("リビングデッドの呼び声")
            or s.has_card_in_hand("強化蘇生")
            or s.has_card_on_szone("強化蘇生"),
        ),
        (
            "card_id:サファイアドラゴン",
            lambda s: s.has_card_in_hand("銀龍の轟咆")
            or s.has_card_on_szone("銀龍の轟咆")
            or s.has_card_in_hand("死者蘇生")
            or s.has_card_on_szone("死者蘇生")
            or s.has_card_in_hand("リビングデッドの呼び声")
            or s.has_card_on_szone("リビングデッドの呼び声")
            or s.has_card_in_hand("強化蘇生")
            or s.has_card_on_szone("強化蘇生"),
        ),
        (
            "card_id:洞窟に潜む竜",
            lambda s: s.has_card_in_hand("戦線復帰") or s.has_card_on_szone("戦線復帰"),
        ),
        (
            "card_id:サファイアドラゴン",
            lambda s: s.has_card_in_hand("戦線復帰") or s.has_card_on_szone("戦線復帰"),
        ),
        ("card_id:洞窟に潜む竜", lambda s: True),
        ("card_id:サファイアドラゴン", lambda s: True),
        ("card_id:アレキサンドライドラゴン", lambda s: True),
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

        if any(command_signature(i) != command_signature(best_command_index) for i in candidate_indices):
            write_debug_log(
                selectable_commands,
                selection_type,
                selection_id,
                f"Ambiguous commands: {[selectable_commands[i] for i in candidate_indices]}",
            )

        best_command_index = random.choice(candidate_indices)

    return best_command_index
