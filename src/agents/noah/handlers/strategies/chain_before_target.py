import random
from typing import Callable

from ygo.constants.enums import PlayerId
from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.chain_before_target import OPTIONS
from ..utils import CARD_MAP, Situation, write_debug_log


def select_chain_before_target(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """チェーンに積まれる前の処理中 (SelectionType:9) & 対象とするカードを選択してください。 (SelectionId:12)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        (
            "card_id:青眼の白龍",
            lambda s: s.rival_lp <= CARD_MAP["青眼の白龍"]["atk"] and s.rival_mzone_count == 0,
        ),
        (
            "card_id:ボマー・ドラゴン",
            lambda s: s.is_rival_turn
            and s.is_battle_phase
            and s.my_mzone_count == 0
            and (
                (
                    s.was_action_taken_by_card("戦線復帰")
                    and s.rival_attacking_monster_atk > CARD_MAP["青眼の白龍"]["def"]
                )
                or (
                    not s.was_action_taken_by_card("戦線復帰")
                    and s.rival_attacking_monster_atk > CARD_MAP["青眼の白龍"]["atk"]
                )
            ),
        ),
        (
            "card_id:ボマー・ドラゴン",
            lambda s: s.rival_monster_atk_max > CARD_MAP["青眼の白龍"]["atk"] and s.rival_face_up_atk_monster_count > 0,
        ),
        (
            "card_id:青眼の白龍",
            lambda s: s.rival_monster_atk_max <= CARD_MAP["青眼の白龍"]["atk"],
        ),
        (
            "card_id:白竜の聖騎士",
            lambda s: (s.has_card_in_deck("青眼の白龍") or s.has_card_in_hand("青眼の白龍"))
            and (
                s.has_card_in_deck("アレキサンドライドラゴン")
                or s.has_card_in_deck("洞窟に潜む竜")
                or s.has_card_in_deck("サファイアドラゴン")
            ),
        ),
        (
            "card_id:白竜の聖騎士",
            lambda s: s.rival_face_down_monster_count > 0,
        ),
        (
            "card_id:創世の竜騎士",
            lambda s: s.has_card_in_deck("青眼の白龍")
            and s.my_hand_monster_count >= 1
            and s.rival_monster_atk_max < CARD_MAP["創世の竜騎士"]["atk"],
        ),
        (
            "card_id:アサルトワイバーン",
            lambda s: s.rival_monster_atk_max < CARD_MAP["アサルトワイバーン"]["atk"]
            and (s.has_card_in_hand("青眼の白龍") or s.has_card_in_grave("青眼の白龍")),
        ),
        (
            "card_id:仮面竜",
            lambda s: s.my_mzone_count == 0 and s.rival_mzone_count >= 1,
        ),
        (
            "card_id:洞窟に潜む竜",
            lambda s: CARD_MAP["洞窟に潜む竜"]["atk"] <= s.rival_monster_atk_max <= CARD_MAP["洞窟に潜む竜"]["def"],
        ),
        (
            "card_id:アレキサンドライドラゴン",
            lambda s: s.rival_monster_atk_max <= CARD_MAP["アレキサンドライドラゴン"]["atk"],
        ),
        (
            "card_id:サファイアドラゴン",
            lambda s: s.rival_monster_atk_max <= CARD_MAP["サファイアドラゴン"]["atk"],
        ),
        (
            "card_id:コドモドラゴン",
            lambda s: s.has_card_in_hand("青眼の白龍"),
        ),
        ("card_id:青眼の白龍", lambda s: True),
        ("card_id:仮面竜", lambda s: True),
        ("card_id:ボマー・ドラゴン", lambda s: True),
        ("card_id:アレキサンドライドラゴン", lambda s: True),
        ("card_id:サファイアドラゴン", lambda s: True),
        ("card_id:白竜の聖騎士", lambda s: True),
        ("card_id:アサルトワイバーン", lambda s: True),
        ("card_id:創世の竜騎士", lambda s: True),
        ("card_id:洞窟に潜む竜", lambda s: True),
        ("card_id:コドモドラゴン", lambda s: True),
        ("card_id:マンジュ・ゴッド", lambda s: True),
        ("card_id:センジュ・ゴッド", lambda s: True),
        ("card_id:ソニックバード", lambda s: True),
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
        # `player_id`選択 (`RIVAL` > `MYSELF`)
        for priority_player in [PlayerId.RIVAL, PlayerId.MYSELF]:
            if filtered_indices := [
                i for i in candidate_indices if selectable_commands[i].player_id == priority_player
            ]:
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
