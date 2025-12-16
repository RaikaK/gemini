from typing import Tuple, List, Optional

from ygo.models.command_request import CommandEntry, CommandRequest
from ygo.models.duel_state_data import DuelStateData
from ygo import constants as c

from src.agents.base_agent import BaseAgent
from src.env.action_data import ActionData
from src.env.state_data import StateData

# ロジックモジュールのインポート
# ※ これらは同じディレクトリ(src/agents/rule/)に配置される想定です
from .atk import AtkExecutor
from .off import OffExecutor
from .chain import ChainExecutor


class RuleAgent(BaseAgent):
    """
    ルールベースエージェント (分割実装版)
    
    atk.py (自分ターン), off.py (相手ターン), chain.py (チェーン処理)
    の3つのモジュールを統括し、状況に応じて適切なロジックに行動選択を委譲します。
    """

    def __init__(self) -> None:
        """
        初期化
        各ロジックExecutorのインスタンスを生成します。
        """
        self.atk_executor = AtkExecutor()
        self.off_executor = OffExecutor()
        self.chain_executor = ChainExecutor()

    def select_action(self, state: StateData) -> Tuple[ActionData, dict | None]:
        """
        ルールベースで行動を選択
        
        Args:
            state: 現在のゲーム状態
            
        Returns:
            選択した行動とその情報
        """
        command_request: CommandRequest = state.command_request
        selectable_commands: List[CommandEntry] = command_request.commands
        
        # 1. 選択肢がない、または1つしかない場合の即時リターン（高速化）
        if not selectable_commands:
            # 基本的にありえないが、念のため
            return ActionData(command_request=command_request, command_entry=None), {"reason": "No commands"}
            
        if len(selectable_commands) == 1:
            return ActionData(command_request=command_request, command_entry=selectable_commands[0]), {"reason": "Only one choice"}

        # 2. 状況分析
        selection_type = command_request.selection_type
        duel_state: DuelStateData = state.duel_state_data
        general_data = duel_state.general_data
        
        is_my_turn = (general_data.which_turn_now == c.PlayerId.MYSELF)
        current_phase = general_data.current_phase
        
        # 選択されたコマンド
        selected_command: Optional[CommandEntry] = None
        reason: str = ""

        # 3. ロジックへの委譲 (Priority Logic)
        # 基本方針: 
        # - チェーン確認 (誘発・フリーチェーン) が最優先
        # - 次にターンプレイヤーごとのメインロジック

        # --- A. チェーン・誘発の判断 (Chain Logic) ---
        # 相手の行動に対するチェーン、あるいは自分の行動に対するチェーン確認画面など
        # コマンドリストに「発動(ACTIVATE)」が含まれ、かつそれが誘発的なタイミングであれば ChainExecutor を優先
        # ※ ここでは簡易的に、チェーン選択画面かどうかを SelectionType やコマンド内容で推論します
        
        # 決定(DECIDE)がある場合（「チェーンしますか？」のダイアログなど）
        # または、相手ターン中に ACTIVATE が可能な場合
        is_chain_situation = self._is_chain_situation(selection_type, selectable_commands, is_my_turn)

        if is_chain_situation:
            selected_command = self.chain_executor.select_action(state, selectable_commands)
            reason = "Chain Logic"

        # --- B. 自分ターンの行動 (Atk Logic) ---
        elif is_my_turn:
            # メインフェイズ、バトルフェイズの行動
            # カード選択(SELECT_CARD)等の処理もAtkExecutorに任せる（攻撃対象選択など）
            selected_command = self.atk_executor.select_action(state, selectable_commands)
            reason = "Atk Logic (My Turn)"

        # --- C. 相手ターンの行動 (Off Logic) ---
        else:
            # 相手ターンのスタンバイ、メイン、エンド等のフリーチェーンタイミング
            selected_command = self.off_executor.select_action(state, selectable_commands)
            reason = "Off Logic (Opponent Turn)"

        # 4. フォールバック (Safety Net)
        # 各Executorが None を返した場合（判断不能時）、安全なデフォルト行動をとる
        if selected_command is None:
            selected_command = self._get_fallback_command(selectable_commands, selection_type)
            reason += " (Fallback)"

        return ActionData(command_request=command_request, command_entry=selected_command), {"selection_reason": reason}

    def _is_chain_situation(self, selection_type: int, commands: List[CommandEntry], is_my_turn: bool) -> bool:
        """
        現在の状況が「チェーン処理（割り込み）」の判断を要するか判定する
        """
        # 明示的なチェーン確認 (Yes/No 等)
        for cmd in commands:
            if cmd.command_type == c.CommandType.DECIDE:
                return True
            if cmd.command_type == c.CommandType.FINALIZE: # チェーン終了選択など
                return True

        # 相手ターンなら、行動できる＝ほぼ全てチェーンかフリーチェーン発動のタイミング
        if not is_my_turn:
            return True
            
        # 自分ターンでも、何かの処理に対する誘発の可能性がある
        # ここは厳密には難しいが、chain.txt のロジックは「相手のアクションに対する反応」が主なので
        # 基本的に ChainExecutor は「相手の行動へのリアクション」として呼び出す
        return False

    def _get_fallback_command(self, commands: List[CommandEntry], selection_type: int) -> CommandEntry:
        """
        ロジックで決定できなかった場合の安全策（デフォルト行動）
        優先順位:
        1. PASS / キャンセル / いいえ
        2. フェイズ移行
        3. リストの先頭
        """
        # PASS (何もしない/ターン終了/チェーンしない) を優先
        for cmd in commands:
            if cmd.command_type == c.CommandType.PASS:
                return cmd
            if cmd.command_type == c.CommandType.DECIDE:
                # 多くのDECIDEは [0]:はい [1]:いいえ の並びが多いが、
                # ここでは安全側に倒すため "No" 的なものを探すか、index 1 を選ぶなどのヒューリスティックが必要
                # 一旦リストの最後（キャンセル寄りであることが多い）を選ぶ
                pass 
        
        # フェイズ移行（メイン→バトル、バトル→メイン2、エンド等）
        for cmd in commands:
            if cmd.command_type == c.CommandType.CHANGE_PHASE:
                return cmd

        # どうしようもない場合は先頭
        return commands[0]

    def update(self, state: StateData, action: ActionData, next_state: StateData, info: dict | None) -> dict | None:
        """
        内部状態の更新
        ルールベースでは学習を行わないため、必要であればログ収集などを行う
        """
        return None