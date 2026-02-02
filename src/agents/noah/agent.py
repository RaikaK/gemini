from datetime import datetime
from pathlib import Path
import pickle

from ygo.models.command_request import CommandEntry, CommandRequest

from src.agents.base_agent import BaseAgent
import src.config as config
from src.env.action_data import ActionData
from src.env.state_data import StateData

from src.agents.noah.decision_maker import DecisionMaker


class Noah(BaseAgent):
    """
    乃亜
    """

    def __init__(self, save_demo: bool = False) -> None:
        """
        初期化する。

        Args:
            save_demo (bool): デモ保存フラグ

        Attributes:
            decision_maker (DecisionMaker): 意思決定ロジック
            save_demo (bool): デモ保存フラグ
            demo_buffer (list): デモバッファ
        """
        # 意思決定ロジック
        self.decision_maker: DecisionMaker = DecisionMaker()

        # デモ保存設定
        self.save_demo: bool = save_demo
        self.demo_buffer: list = []

        if save_demo:
            try:
                config.RULE_DEMONSTRATION_DIR.mkdir(parents=True, exist_ok=True)

            except Exception:
                self.save_demo = False

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        try:
            command_request: CommandRequest = state.command_request
            selectable_commands: list[CommandEntry] = command_request.commands

            # 行動選択
            command_index: int = self.decision_maker.select_command_index(state, selectable_commands)

            if command_index not in range(len(selectable_commands)):
                command_index = 0

            # 行動作成
            selected_command: CommandEntry = selectable_commands[command_index]
            action: ActionData = ActionData(command_request=command_request, command_entry=selected_command)

            return action, None

        except Exception:
            return ActionData(command_request=command_request, command_entry=selectable_commands[0]), None

    def update(self, state: StateData, action: ActionData, next_state: StateData, info: dict | None) -> dict | None:
        if not self.save_demo:
            return None

        try:
            # デモ追加
            self.demo_buffer.append(
                {
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "info": info,
                }
            )

            # デモ保存
            if next_state.is_duel_end:
                self.save_demonstration()
                self.demo_buffer.clear()

        except Exception:
            pass

        return None

    def save_demonstration(self) -> None:
        """
        デモを保存する。
        """
        try:
            # ファイル名作成
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename: Path = config.RULE_DEMONSTRATION_DIR / f"{timestamp}.pkl"

            # デモ保存
            with open(filename, "wb") as f:
                pickle.dump(self.demo_buffer, f)

        except Exception:
            pass
