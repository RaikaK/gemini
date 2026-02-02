from ygo.constants.enums import SelectionType
from ygo.constants.selection_id import SelectionId
from ygo.models.command_request import CommandEntry

from src.env.state_data import StateData

import src.agents.noah.handlers.strategies as strategies
from src.agents.noah.handlers.utils.fallback import select_random_command_index


class DecisionMaker:
    """
    意思決定ロジック
    """

    def __init__(self) -> None:
        """
        初期化する。
        """

    def select_command_index(self, state: StateData, selectable_commands: list[CommandEntry]) -> int:
        """
        行動のインデックスを選択する。

        Args:
            state (StateData): 状態データ
            selectable_commands (list[CommandEntry]): 選択可能な行動リスト

        Returns:
            int: 選択した行動のインデックス
        """
        try:
            selection_type: int | None = None
            selection_id: int | None = None
            command_index: int = 0

            # 行動要求の種類/説明を取得
            selection_type = state.command_request.selection_type
            selection_id = state.command_request.selection_id

            # ドローフェイズ
            if selection_type == SelectionType.DRAW_PHASE:
                # 値無し
                if selection_id == SelectionId.NO_VALUE:
                    if len(selectable_commands) == 1:
                        command_index = 0

                    else:
                        raise ValueError("Multiple commands available in Draw Phase")

                else:
                    command_index = select_random_command_index(selectable_commands, selection_type, selection_id)

            # メインフェイズ
            elif selection_type == SelectionType.MAIN_PHASE:
                # 値無し
                if selection_id == SelectionId.NO_VALUE:
                    command_index = strategies.select_main_phase(
                        state, selectable_commands, selection_type, selection_id
                    )

                else:
                    command_index = select_random_command_index(selectable_commands, selection_type, selection_id)

            # バトルフェイズ
            elif selection_type == SelectionType.BATTLE_PHASE:
                # 値無し
                if selection_id == SelectionId.NO_VALUE:
                    command_index = strategies.select_battle_phase(
                        state, selectable_commands, selection_type, selection_id
                    )

                else:
                    command_index = select_random_command_index(selectable_commands, selection_type, selection_id)

            # 攻撃対象選択
            elif selection_type == SelectionType.SELECT_ATTACK_TARGET:
                # 値無し
                if selection_id == SelectionId.NO_VALUE:
                    command_index = strategies.select_attack_target(
                        state, selectable_commands, selection_type, selection_id
                    )

                else:
                    command_index = select_random_command_index(selectable_commands, selection_type, selection_id)

            # 発動確認
            elif selection_type == SelectionType.CHECK_ACTIVATION:
                # 値無し
                if selection_id == SelectionId.NO_VALUE:
                    command_index = strategies.select_activate_confirmation(
                        state, selectable_commands, selection_type, selection_id
                    )

                else:
                    command_index = select_random_command_index(selectable_commands, selection_type, selection_id)

            # モンスターの召喚中
            elif selection_type == SelectionType.SUMMONING:
                # リリースするカードを選択してください。
                if selection_id == SelectionId.SELECT_CARD_AS_TRIBUTE:
                    command_index = strategies.select_summon_release(
                        state, selectable_commands, selection_type, selection_id
                    )

                else:
                    command_index = select_random_command_index(selectable_commands, selection_type, selection_id)

            # チェーンに積まれる前の処理中
            elif selection_type == SelectionType.CHAIN_SETTING:
                # 手札のカードを墓地へ送ってください。
                if selection_id == SelectionId.SEND_A_CARD_IN_YOUR_HAND_TO_THE_GRAVEYARD:
                    command_index = strategies.select_chain_before_send_to_grave(
                        state, selectable_commands, selection_type, selection_id
                    )

                # 対象とするカードを選択してください。
                elif selection_id == SelectionId.SELECT_CARD_TO_TARGET:
                    command_index = strategies.select_chain_before_target(
                        state, selectable_commands, selection_type, selection_id
                    )

                # モンスターを選択してください。
                elif selection_id == SelectionId.SELECT_MONSTER:
                    command_index = strategies.select_chain_before_monster(
                        state, selectable_commands, selection_type, selection_id
                    )

                # 破壊するカードを選択してください。
                elif selection_id == SelectionId.SELECT_CARD_TO_DESTROY:
                    command_index = strategies.select_chain_before_destroy(
                        state, selectable_commands, selection_type, selection_id
                    )

                # 手札を捨ててください。
                elif selection_id == SelectionId.DISCARD_FROM_YOUR_HAND:
                    command_index = strategies.select_chain_before_discard(
                        state, selectable_commands, selection_type, selection_id
                    )

                # 裏側守備表示にするモンスターを選択してください。
                elif selection_id == SelectionId.SELECT_MONSTER_TO_SWITCH_TO_FACEDOWN_DEFENSE_POSITION:
                    command_index = strategies.select_chain_before_face_down(
                        state, selectable_commands, selection_type, selection_id
                    )

                else:
                    command_index = select_random_command_index(selectable_commands, selection_type, selection_id)

            # チェーンの効果処理中
            elif selection_type == SelectionType.CHAIN_RUNNING:
                # 手札から魔法・罠カードをセットしますか？
                if selection_id == SelectionId.SET_A_SPELL_OR_TRAP_CARD_ON_THE_FIELD_Q:
                    command_index = strategies.select_chain_effect_set_query(
                        state, selectable_commands, selection_type, selection_id
                    )

                # 手札からセットする魔法・罠カードを選択してください。
                elif selection_id == SelectionId.SELECT_SPELL_OR_TRAP_CARD_TO_SET_ON_FIELD:
                    command_index = strategies.select_chain_effect_set_card(
                        state, selectable_commands, selection_type, selection_id
                    )

                # デッキから特殊召喚するモンスターを選択してください。
                elif selection_id == SelectionId.SELECT_MONSTER_TO_SPECIAL_SUMMON_FROM_YOUR_DECK:
                    command_index = strategies.select_chain_effect_special_summon_deck(
                        state, selectable_commands, selection_type, selection_id
                    )

                # デッキから手札に加えるカードを選択してください。
                elif selection_id == SelectionId.SELECT_CARD_TO_ADD_FROM_YOUR_DECK_TO_YOUR_HAND:
                    command_index = strategies.select_chain_effect_search_deck(
                        state, selectable_commands, selection_type, selection_id
                    )

                # 墓地へ送るカードを選択してください。
                elif selection_id == SelectionId.SELECT_CARD_TO_SEND_TO_GRAVEYARD:
                    command_index = strategies.select_chain_effect_send_to_grave(
                        state, selectable_commands, selection_type, selection_id
                    )

                # 儀式召喚に必要なレベル分のモンスターを選択してください。
                elif selection_id == SelectionId.SELECT_NECESSARY_MONSTER_TO_MATCH_REQUIRED_NUMBER_OF_LEVEL:
                    command_index = strategies.select_chain_effect_ritual_select(
                        state, selectable_commands, selection_type, selection_id
                    )

                # 儀式召喚に必要なレベル分のモンスターをリリースしてください。
                elif selection_id == SelectionId.TRIBUTE_NECESSARY_MONSTER_TO_MATCH_REQUIRED_NUMBER_OF_LEVEL:
                    command_index = strategies.select_chain_effect_ritual_release(
                        state, selectable_commands, selection_type, selection_id
                    )

                # 表示形式を選択してください。
                elif selection_id == SelectionId.SELECT_BATTLE_POSITION_OF_CARD:
                    command_index = strategies.select_chain_effect_position(
                        state, selectable_commands, selection_type, selection_id
                    )

                # 特殊召喚するモンスターを選択してください。
                elif selection_id == SelectionId.SELECT_MONSTER_TO_SPECIAL_SUMMON:
                    command_index = strategies.select_chain_effect_special_summon(
                        state, selectable_commands, selection_type, selection_id
                    )

                # 特殊召喚するモンスターを手札から選択してください。
                elif selection_id == SelectionId.SELECT_MONSTER_FROM_HAND_TO_SPECIAL_SUMMON:
                    command_index = strategies.select_chain_effect_special_summon_hand(
                        state, selectable_commands, selection_type, selection_id
                    )

                else:
                    command_index = select_random_command_index(selectable_commands, selection_type, selection_id)

            # その他
            elif selection_type == SelectionType.OTHER:
                # 戦闘が巻き戻されました。攻撃を続けますか？
                if selection_id == SelectionId.CONTINUE_TO_ATTACK_Q:
                    command_index = strategies.select_other_rewind_attack(
                        state, selectable_commands, selection_type, selection_id
                    )

                else:
                    command_index = select_random_command_index(selectable_commands, selection_type, selection_id)

            else:
                command_index = select_random_command_index(selectable_commands, selection_type, selection_id)

            return command_index

        except Exception as error:
            return select_random_command_index(selectable_commands, selection_type, selection_id, error=error)
