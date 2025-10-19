from ygo.models.command_request import CommandEntry, CommandRequest


class ActionData:
    """
    行動データ
    """

    def __init__(self, command_request: CommandRequest, command_entry: CommandEntry) -> None:
        """
        初期化する。

        Args:
            command_request (CommandRequest): 行動要求に関する情報
            command_entry (CommandEntry): 選択された行動

        Attributes:
            command_request (CommandRequest): 行動要求に関する情報
            command_entry (CommandEntry): 選択された行動
            command_index (int): 選択された行動のインデックス
        """
        self.command_request: CommandRequest = command_request
        self.command_entry: CommandEntry = command_entry
        self.command_index: int = self.get_command_index()

    def get_command_index(self) -> int:
        """
        選択された行動のインデックスを返す。

        Returns:
            int: 選択された行動のインデックス
        """
        for index, command in enumerate(self.command_request.commands):
            if self.command_entry == command:
                return index

        raise ValueError("The CommandEntry does not exist in the CommandRequest's commands.")
