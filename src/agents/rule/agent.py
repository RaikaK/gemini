from ygo.models.command_request import CommandEntry, CommandRequest
from ygo.models.duel_state_data import DuelStateData
from ygo.models.duel_card import DuelCard
from ygo.models.general_data import GeneralData
from ygo import constants as c

from src.agents.base_agent import BaseAgent
from src.env.action_data import ActionData
from src.env.state_data import StateData


class RuleAgent(BaseAgent):
    """
    ルールベースエージェント
    
    以下の優先順位で行動を選択：
    1. 効果発動（状況に応じて）
    2. 攻撃（攻撃力が高い順）
    3. 召喚・特殊召喚（攻撃力が高い順）
    4. 表示形式変更（守備→攻撃）
    5. セット
    6. フェイズ移行・パス
    """

    def __init__(self) -> None:
        """初期化"""
        pass

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        """
        ルールベースで行動を選択
        
        Args:
            state: 現在のゲーム状態
            
        Returns:
            選択した行動とその情報
        """
        command_request: CommandRequest = state.command_request
        selectable_commands: list[CommandEntry] = command_request.commands
        duel_state: DuelStateData = state.duel_state_data
        
        # 選択肢が1つしかない場合はそれを選択
        if len(selectable_commands) == 1:
            return ActionData(command_request=command_request, command_entry=selectable_commands[0]), None
        
        # ルールベースで最適な行動を選択
        selected_command = self._select_best_command(selectable_commands, duel_state, command_request)
        action: ActionData = ActionData(command_request=command_request, command_entry=selected_command)
        
        return action, {"selection_reason": f"CommandType: {selected_command.command_type}"}

    def _select_best_command(
        self, 
        commands: list[CommandEntry], 
        duel_state: DuelStateData,
        command_request: CommandRequest
    ) -> CommandEntry:
        """
        最適な行動を選択
        
        Args:
            commands: 選択可能な行動リスト
            duel_state: デュエル状態
            command_request: 行動要求情報
            
        Returns:
            選択した行動
        """
        general_data: GeneralData = duel_state.general_data
        duel_cards: list[DuelCard] = duel_state.duel_card_table
        
        # 優先度1: 効果の発動（攻撃的な状況で）
        if self._should_activate_effect(general_data, duel_cards):
            activate_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.ACTIVATE]
            if activate_commands:
                return activate_commands[0]
        
        # 優先度2: 攻撃（攻撃力が高い順）
        attack_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.ATTACK]
        if attack_commands:
            return self._select_best_attack(attack_commands, duel_cards)
        
        # 優先度3: 召喚（攻撃力が高い順）
        summon_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.SUMMON]
        if summon_commands:
            return self._select_best_summon(summon_commands, duel_cards)
        
        # 優先度4: 特殊召喚（攻撃力が高い順）
        sp_summon_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.SUMMON_SP]
        if sp_summon_commands:
            return self._select_best_summon(sp_summon_commands, duel_cards)
        
        # 優先度5: 反転召喚
        reverse_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.REVERSE]
        if reverse_commands:
            return self._select_best_summon(reverse_commands, duel_cards)
        
        # 優先度6: 守備→攻撃表示変更（攻撃力が高い順）
        turn_atk_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.TURN_ATK]
        if turn_atk_commands:
            return self._select_best_position_change(turn_atk_commands, duel_cards)
        
        # 優先度7: 魔法・罠のセット（手札を減らす）
        set_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.SET]
        if set_commands:
            return set_commands[0]
        
        # 優先度8: モンスターのセット（攻撃力が低いもの）
        set_monst_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.SET_MONST]
        if set_monst_commands:
            return self._select_weakest_for_set(set_monst_commands, duel_cards)
        
        # 優先度9: ペンデュラムスケールに発動
        pendulum_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.PENDULUM]
        if pendulum_commands:
            return pendulum_commands[0]
        
        # 優先度10: 効果適用（残っている場合）
        apply_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.APPLY]
        if apply_commands:
            return apply_commands[0]
        
        # 優先度11: ドロー
        draw_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.DRAW]
        if draw_commands:
            return draw_commands[0]
        
        # 優先度12: 決定・選択終了
        decide_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.DECIDE]
        if decide_commands:
            return decide_commands[0]
        
        finalize_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.FINALIZE]
        if finalize_commands:
            return finalize_commands[0]
        
        # 優先度13: フェイズ移行（バトルフェイズを優先）
        phase_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.CHANGE_PHASE]
        if phase_commands:
            return self._select_best_phase_change(phase_commands, general_data)
        
        # 優先度14: パス
        pass_commands = [cmd for cmd in commands if cmd.command_type == c.CommandType.PASS]
        if pass_commands:
            return pass_commands[0]
        
        # デフォルト: 最初のコマンド
        return commands[0]
    
    def _should_activate_effect(self, general_data: GeneralData, duel_cards: list[DuelCard]) -> bool:
        """
        効果を発動すべきか判断
        
        Args:
            general_data: 一般情報
            duel_cards: デュエルカード一覧
            
        Returns:
            効果を発動すべきならTrue
        """
        # 基本的に効果は発動する方針
        # ただし、相手のLPが低い場合や自分のフィールドが強い場合は積極的に
        my_lp = general_data.lp[0]
        rival_lp = general_data.lp[1]
        
        # 相手のLPが少ない、または自分のLPが多い場合は効果を発動
        return rival_lp < 4000 or my_lp > 6000
    
    def _select_best_attack(self, attack_commands: list[CommandEntry], duel_cards: list[DuelCard]) -> CommandEntry:
        """
        最適な攻撃を選択（攻撃力が高い順）
        
        Args:
            attack_commands: 攻撃コマンドリスト
            duel_cards: デュエルカード一覧
            
        Returns:
            選択した攻撃コマンド
        """
        # 攻撃力が高い順にソート
        sorted_commands = sorted(
            attack_commands,
            key=lambda cmd: self._get_card_attack_power(cmd, duel_cards),
            reverse=True
        )
        return sorted_commands[0]
    
    def _select_best_summon(self, summon_commands: list[CommandEntry], duel_cards: list[DuelCard]) -> CommandEntry:
        """
        最適な召喚を選択（攻撃力が高い順）
        
        Args:
            summon_commands: 召喚コマンドリスト
            duel_cards: デュエルカード一覧
            
        Returns:
            選択した召喚コマンド
        """
        # 攻撃力が高い順にソート
        sorted_commands = sorted(
            summon_commands,
            key=lambda cmd: self._get_card_attack_power(cmd, duel_cards),
            reverse=True
        )
        return sorted_commands[0]
    
    def _select_best_position_change(self, turn_commands: list[CommandEntry], duel_cards: list[DuelCard]) -> CommandEntry:
        """
        最適な表示形式変更を選択（攻撃力が高い順）
        
        Args:
            turn_commands: 表示形式変更コマンドリスト
            duel_cards: デュエルカード一覧
            
        Returns:
            選択した表示形式変更コマンド
        """
        # 攻撃力が高い順にソート
        sorted_commands = sorted(
            turn_commands,
            key=lambda cmd: self._get_card_attack_power(cmd, duel_cards),
            reverse=True
        )
        return sorted_commands[0]
    
    def _select_weakest_for_set(self, set_commands: list[CommandEntry], duel_cards: list[DuelCard]) -> CommandEntry:
        """
        セット用に最弱のモンスターを選択（攻撃力が低い順）
        
        Args:
            set_commands: セットコマンドリスト
            duel_cards: デュエルカード一覧
            
        Returns:
            選択したセットコマンド
        """
        # 攻撃力が低い順にソート
        sorted_commands = sorted(
            set_commands,
            key=lambda cmd: self._get_card_attack_power(cmd, duel_cards),
            reverse=False
        )
        return sorted_commands[0]
    
    def _select_best_phase_change(self, phase_commands: list[CommandEntry], general_data: GeneralData) -> CommandEntry:
        """
        最適なフェイズ移行を選択
        
        Args:
            phase_commands: フェイズ移行コマンドリスト
            general_data: 一般情報
            
        Returns:
            選択したフェイズ移行コマンド
        """
        # バトルフェイズがあれば優先
        for cmd in phase_commands:
            if cmd.phase == c.Phase.BATTLE:
                return cmd
        
        # なければ最初のフェイズ移行
        return phase_commands[0]
    
    def _get_card_attack_power(self, command: CommandEntry, duel_cards: list[DuelCard]) -> int:
        """
        コマンドに関連するカードの攻撃力を取得
        
        Args:
            command: コマンド
            duel_cards: デュエルカード一覧
            
        Returns:
            攻撃力（不明の場合は0）
        """
        # table_indexからカード情報を取得
        if command.table_index >= 0 and command.table_index < len(duel_cards):
            card = duel_cards[command.table_index]
            return card.atk_val if card.atk_val >= 0 else 0
        
        # card_idから推定（手札の場合など）
        # 実際のカード情報がない場合はcard_idを返す（大きいほど強いと仮定）
        return command.card_id if command.card_id > 0 else 0

    def update(self, state: StateData, action: ActionData, next_state: StateData, info: dict | None) -> dict | None:
        """
        内部状態の更新（ルールベースでは不要）
        
        Args:
            state: 現在の状態
            action: 実行した行動
            next_state: 次の状態
            info: 追加情報
            
        Returns:
            更新情報（なし）
        """
        return None
