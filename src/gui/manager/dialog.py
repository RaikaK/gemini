import tkinter as tk

from ygo.gui.manager.const import Const
from ygo.gui.manager.dialog_manager import DialogManager

from src.gui.frame import GUIFrame


class GUIDialog(DialogManager):
    """
    GUIダイアログ
    """

    def __init__(self, udi_gui_frame: GUIFrame, master: tk.Misc, **key) -> None:
        """
        初期化する。
        """
        super().__init__(udi_gui_frame, master, **key)

        factor: float = self.udi_gui_frame.factor
        scaled_font_size: int = int(int(Const.DIALOG_FONT[1]) * factor)
        scaled_font: tuple = (Const.DIALOG_FONT[0], scaled_font_size, Const.DIALOG_FONT[2])
        scaled_wrap_length: int = int(Const.DIALOG_WRAP_LENGTH * factor)
        self.text_label.config(font=scaled_font, wraplength=scaled_wrap_length)
