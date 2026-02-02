from datetime import datetime
from pathlib import Path
import random

from ygo.models.command_request import CommandEntry

import src.config as config


def write_debug_log(
    selectable_commands: list[CommandEntry],
    selection_type: int | None,
    selection_id: int | None,
    error: str | Exception | None = None,
) -> None:
    """
    デバッグ情報をファイルに書き込む。

    Args:
        selectable_commands (list[CommandEntry]): 選択可能な行動リスト
        selection_type (int | None): 行動要求の種類
        selection_id (int | None): 行動要求の説明
        error (str | Exception | None): 発生した例外
    """
    # ログの設定
    log_file_path: Path = config.PROJECT_ROOT / config.DEBUG_FILE
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # メッセージの作成
    if error is None:
        # 未定義の行動要求
        msg = f"Undefined: SelectionType={selection_type}, SelectionId={selection_id}"

    else:
        # 例外発生
        msg = f"Error: SelectionType={selection_type}, SelectionId={selection_id}, Content={str(error)}"

    commands_msg: str = "|".join([str(command) for command in selectable_commands])
    msg = f"{msg}, Commands=[{commands_msg}]"
    log_entry: str = f"[{timestamp}] {msg}\n"

    # ファイルに追記
    try:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    except Exception:
        pass


def select_random_command_index(
    selectable_commands: list[CommandEntry],
    selection_type: int | None,
    selection_id: int | None,
    error: Exception | None = None,
) -> int:
    """
    ランダムに行動のインデックスを選択する。

    Args:
        selectable_commands (list[CommandEntry]): 選択可能な行動リスト
        selection_type (int | None): 行動要求の種類
        selection_id (int | None): 行動要求の説明
        error (Exception | None): 発生した例外

    Returns:
        int: 選択した行動のインデックス
    """
    # 選択可能な行動が無い場合
    if not selectable_commands:
        return 0

    # ログ出力
    write_debug_log(selectable_commands, selection_type, selection_id, error)

    # ランダムに選択
    try:
        return random.randrange(len(selectable_commands))

    except Exception:
        return 0
