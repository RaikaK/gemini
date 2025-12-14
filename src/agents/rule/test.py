# test.py
from __future__ import annotations

from ygo.models.command_request import CommandEntry, CommandRequest

from src.agents.base_agent import BaseAgent
from src.env.action_data import ActionData
from src.env.state_data import StateData


class DoNothingAgent(BaseAgent):
    """
    超簡易ルールベース：
    何があっても「できるだけ何もしない」選択肢を選ぶ。

    ※ 環境仕様上、is_cmd_required=True のときは CommandEntry を必ず1つ返す必要があるため、
       No/フェイズ進行/先頭 といった “無害寄り” の選択で擬似的に何もしないを実現する。
    """

    def __init__(self) -> None:
        pass

    def select_action(self, state: StateData) -> tuple[ActionData, dict | None]:
        cr: CommandRequest = state.command_request
        cmds: list[CommandEntry] = cr.commands

        if not cmds:
            raise RuntimeError("No commands in CommandRequest (commands is empty).")

        chosen = self._pick_most_passive(cmds)
        return ActionData(command_request=cr, command_entry=chosen), None

    def update(
        self,
        state: StateData,
        action: ActionData,
        next_state: StateData,
        info: dict | None,
    ) -> dict | None:
        return None

    @staticmethod
    def _pick_most_passive(cmds: list[CommandEntry]) -> CommandEntry:
        """
        “何もしない寄り” を選ぶ簡易ヒューリスティック。
        constantsの具体値が不明でも動くように、フィールド値で安全に判断する。

        優先：
        1) Yes/No があるなら「Noっぽい」(0) を優先
        2) フェイズ遷移があるなら、一番後ろへ進むものを優先（早く終わりやすい）
        3) それ以外は先頭
        """
        # 1) Yes/No の選択肢 → まず "No" を優先（YesNo enum が不明なので 0 をNo扱い）
        no_candidates = [c for c in cmds if hasattr(c, "yes_no") and int(c.yes_no) == 0]
        if no_candidates:
            return no_candidates[0]

        # 2) フェイズ遷移がある → できるだけ後ろのフェイズへ
        phase_cmds = [c for c in cmds if hasattr(c, "phase") and int(c.phase) != 0]
        if phase_cmds:
            return max(phase_cmds, key=lambda x: int(x.phase))

        # 3) 最後の手段：先頭
        return cmds[0]
