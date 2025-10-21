import tkinter as tk

from ygo import models as mdl
from ygo.gui.manager.const import Const
from ygo.gui.manager.log_manager import LogLabel, LogManager
from ygo.util.text import TextUtil


class GUILogLabel(LogLabel):
    """
    GUIログラベル
    """

    def __init__(self, master: tk.Misc, num: int, text: str, udi_gui_frame, factor: float) -> None:
        """
        初期化する。
        """
        scaled_bd: int = max(1, int(Const.LOG_BD * factor))
        scaled_num_font: tuple = (
            Const.LOG_NUM_FONT[0],
            max(7, int(int(Const.LOG_NUM_FONT[1]) * factor)),
            Const.LOG_NUM_FONT[2],
        )
        scaled_text_font: tuple = (
            Const.LOG_TEXT_FONT[0],
            max(7, int(int(Const.LOG_TEXT_FONT[1]) * factor)),
        )
        scaled_wrap_length: int = int(Const.LOG_WRAP_LENGTH * factor)

        tk.Frame.__init__(self, master, relief=tk.SOLID, bd=scaled_bd)

        self.text = text
        self.udi_gui_frame = udi_gui_frame

        num_label = tk.Label(self, text=str(num), font=scaled_num_font)
        num_label.pack()

        text_label = tk.Label(self, text=self.text, font=scaled_text_font, wraplength=scaled_wrap_length)
        text_label.pack()


class GUILog(LogManager):
    """
    GUIログ
    """

    def update(self, udi_io_duel_log: list[mdl.DuelLogDataEntry]) -> None:
        """
        更新する。
        """
        factor: float = self.udi_gui_frame.factor

        text_util: TextUtil = self.udi_gui_frame.text_util

        if len(self.label_list) < len(udi_io_duel_log):
            scaled_pady: int = max(1, int(Const.LOG_PADY * factor))

            for i in range(len(self.label_list), len(udi_io_duel_log)):
                log: mdl.DuelLogDataEntry = udi_io_duel_log[i]
                text: str = text_util.get_duel_log_entry_text(log)
                label: GUILogLabel = GUILogLabel(self.frame, i, text, self.udi_gui_frame, factor)
                label.pack(pady=scaled_pady)

                for child in label.children.values():
                    child.bind("<MouseWheel>", self._on_mousewheel)

                label.bind("<MouseWheel>", self._on_mousewheel)
                self.label_list.append(label)

        elif len(self.label_list) > len(udi_io_duel_log):
            for i in range(len(udi_io_duel_log), len(self.label_list)):
                self.label_list[i].destroy()
            self.label_list = self.label_list[: len(udi_io_duel_log)]
