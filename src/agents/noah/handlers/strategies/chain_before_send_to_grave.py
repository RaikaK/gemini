import random
from typing import Callable

from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

from ..options.chain_before_send_to_grave import OPTIONS
from ..utils import CARD_MAP, Situation, write_debug_log


def select_chain_before_send_to_grave(
    state: StateData, selectable_commands: list[CommandEntry], selection_type: int, selection_id: int
) -> int:
    """チェーンに積まれる前の処理中 (SelectionType:9) & 手札のカードを墓地へ送ってください。 (SelectionId:7)"""

    # 状況取得
    situation: Situation = Situation(state)

    # ランキング定義
    ranking: list[tuple[str, Callable[[Situation], bool]]] = [
        (
            "card_id:青眼の白龍",
            lambda s: s.has_card_in_hand("死者蘇生")
            or s.has_card_in_hand("銀龍の轟咆")
            or s.has_card_on_szone("銀龍の轟咆")
            or s.has_card_in_hand("戦線復帰")
            or s.has_card_on_szone("戦線復帰")
            or s.has_card_on_szone("リビングデッドの呼び声")
            or s.has_card_in_hand("早すぎた埋葬"),
        ),
        (
            "card_id:アレキサンドライドラゴン",
            lambda s: (s.has_card_in_hand("強化蘇生") or s.has_card_on_szone("強化蘇生"))
            and not s.has_card_in_grave("アレキサンドライドラゴン"),
        ),
        (
            "card_id:洞窟に潜む竜",
            lambda s: (s.has_card_in_hand("戦線復帰") or s.has_card_on_szone("戦線復帰"))
            and not s.has_card_in_grave("洞窟に潜む竜"),
        ),
        (
            "card_id:白竜の聖騎士",
            lambda s: (
                s.has_card_in_hand("死者蘇生")
                or s.has_card_on_szone("リビングデッドの呼び声")
                or s.has_card_on_szone("戦線復帰")
            )
            and not s.has_card_in_hand("白竜降臨")
            and not s.has_card_in_hand("高等儀式術"),
        ),
        (
            "card_id:青眼の白龍",
            lambda s: s.has_card_on_mzone("アサルトワイバーン")
            and (
                s.rival_monster_atk_min < CARD_MAP["アサルトワイバーン"]["atk"]
                or s.rival_face_up_def_monster_def_min < CARD_MAP["アサルトワイバーン"]["atk"]
            ),
        ),
        (
            "card_id:白竜の聖騎士",
            lambda s: s.has_card_on_mzone("アサルトワイバーン")
            and (
                s.rival_monster_atk_min < CARD_MAP["アサルトワイバーン"]["atk"]
                or s.rival_face_up_def_monster_def_min < CARD_MAP["アサルトワイバーン"]["atk"]
            ),
        ),
        ("card_id:白竜の聖騎士", lambda s: not s.has_card_in_deck("青眼の白龍")),
        (
            "card_id:白竜の聖騎士",
            lambda s: not s.has_card_in_hand("白竜降臨")
            and not s.has_card_in_deck("白竜降臨")
            and not s.has_card_in_hand("高等儀式術")
            and not s.has_card_in_deck("高等儀式術"),
        ),
        ("card_id:センジュ・ゴッド", lambda s: not s.has_card_in_deck("白竜の聖騎士")),
        (
            "card_id:マンジュ・ゴッド",
            lambda s: not s.has_card_in_deck("白竜の聖騎士")
            and not s.has_card_in_deck("白竜降臨")
            and not s.has_card_in_deck("高等儀式術"),
        ),
        (
            "card_id:ソニックバード",
            lambda s: not s.has_card_in_deck("白竜降臨") and not s.has_card_in_deck("高等儀式術"),
        ),
        (
            "card_id:白竜の聖騎士",
            lambda s: not s.has_card_in_hand("白竜降臨")
            and not s.has_card_in_hand("高等儀式術")
            and s.my_deck_count < 10,
        ),
        (
            "card_id:白竜降臨",
            lambda s: s.has_face_card_on_mzone("白竜の聖騎士")
            and not s.has_card_in_hand("白竜の聖騎士")
            and not s.has_card_in_deck("白竜の聖騎士"),
        ),
        (
            "card_id:アレキサンドライドラゴン",
            lambda s: s.my_monster_atk_max >= 2000 and s.has_card_in_hand("アレキサンドライドラゴン"),
        ),
        ("card_id:洞窟に潜む竜", lambda s: s.is_lp_advantage),
        ("card_id:アレキサンドライドラゴン", lambda s: s.my_hand_count >= 6),
        ("card_id:洞窟に潜む竜", lambda s: s.lp_diff >= 3000 or s.my_face_up_atk_monster_count >= 2),
        ("card_id:砂塵の大竜巻", lambda s: s.rival_szone_count == 0 and s.rival_hand_count == 0),
        (
            "card_id:収縮",
            lambda s: s.my_monster_atk_max >= s.rival_monster_atk_max + 2000 and s.rival_monster_atk_max > 0,
        ),
        (
            "card_id:禁じられた聖槍",
            lambda s: s.rival_face_down_spell_count == 0 and s.rival_hand_count == 0 and s.my_mzone_count == 0,
        ),
        (
            "card_id:大嵐",
            lambda s: s.rival_szone_count == 0 and s.is_lp_advantage and s.my_face_down_spell_count >= 3,
        ),
        ("card_id:激流葬", lambda s: s.rival_mzone_count == 0 and s.my_lp >= 7500),
        ("card_id:洞窟に潜む竜", lambda s: True),
        ("card_id:アレキサンドライドラゴン", lambda s: True),
        ("card_id:収縮", lambda s: True),
        ("card_id:砂塵の大竜巻", lambda s: True),
        ("card_id:センジュ・ゴッド", lambda s: True),
        ("card_id:ソニックバード", lambda s: True),
        ("card_id:マンジュ・ゴッド", lambda s: True),
        ("card_id:白竜降臨", lambda s: True),
        ("card_id:禁じられた聖槍", lambda s: True),
        ("card_id:大嵐", lambda s: True),
        ("card_id:激流葬", lambda s: True),
        ("card_id:強化蘇生", lambda s: True),
        ("card_id:銀龍の轟咆", lambda s: True),
        ("card_id:白竜の聖騎士", lambda s: True),
        ("card_id:青眼の白龍", lambda s: True),
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
