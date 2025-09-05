import sys

sys.path.append("C:/Users/b1/Desktop/master-duel-ai")

from ygo.models.duel_state_data import DuelStateData
from ygo.models.command_request import CommandRequest, CommandEntry


class ActionData:
    def __init__(
        self,
        state: DuelStateData,
        command_request: CommandRequest,
        command_entry: CommandEntry,
    ):
        """ある状態sと行動aをまとめたデータ"""
        self.state: DuelStateData = state
        self.command_request: CommandRequest = command_request
        self.command_entry: CommandEntry = command_entry
        self.command_index: int = self.get_command_index()

    def get_command_index(self) -> int:
        """command_entryがcommand_requestのcommandsの何番目にあるかを返す"""
        for i, cmd in enumerate(self.command_request.commands):
            if self.command_entry == cmd:
                return i

        # command_entryがcommandsに存在しない場合はエラー
        raise ValueError(
            "指定されたCommandEntryは、CommandRequestのcommandsに存在しません。"
        )
