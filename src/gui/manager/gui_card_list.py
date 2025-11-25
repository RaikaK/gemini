import tkinter as tk

from ygo import constants as c
from ygo import models as mdl
from ygo.gui.manager.card_list_manager import CardLabel, CardListManager
from ygo.gui.manager.const import Const
from ygo.gui.manager.scollable_frame import ScrollableFrameY


class GUICardListLabel(CardLabel):
    """
    GUIカードリストラベル
    """

    def __init__(self, master: tk.Misc, udi_gui_frame, factor: float) -> None:
        """
        初期化する。
        """
        super().__init__(master, udi_gui_frame)

        scaled_bd: int = max(1, int(Const.C_LIST_BD * factor))
        scaled_font: tuple = (
            Const.C_LIST_FONT[0],
            max(8, int(int(Const.C_LIST_FONT[1]) * factor)),
        )
        scaled_wrap_length: int = int(Const.C_LIST_WRAP_LENGTH * factor)

        self.config(bd=scaled_bd)
        self.info_label.config(font=scaled_font, wraplength=scaled_wrap_length)


class GUICardList(CardListManager):
    """
    GUIカードリスト
    """

    def __init__(self, udi_gui_frame, master: tk.Misc, **key) -> None:
        """
        初期化する。
        """
        factor: float = udi_gui_frame.factor

        self.udi_gui_frame = udi_gui_frame
        self.master: tk.Misc = master
        self.key: dict = key

        self.player_id: int = c.enums.PlayerId.NO_VALUE
        self.pos_id: int = c.enums.PosId.NO_VALUE
        self.duel_card_table: list[mdl.DuelCard] | None = None

        scaled_info_font: tuple = (
            Const.C_LIST_INFO_FONT[0],
            max(8, int(int(Const.C_LIST_INFO_FONT[1]) * factor)),
            Const.C_LIST_INFO_FONT[2],
        )
        scaled_info_wrap_length: int = int(Const.C_LIST_INFO_WRAP_LENGTH * factor)

        self.info_label: tk.Label = tk.Label(
            self.master,
            text=f"P{self.player_id}:{c.enums.PosId(self.pos_id)}",
            font=scaled_info_font,
            wraplength=scaled_info_wrap_length,
        )
        self.info_label.pack()

        ScrollableFrameY.__init__(self, master, **key)

        self.card_list: list[GUICardListLabel] = []
        MAX_CARD_NUM = 60 + 15

        scaled_padx: int = max(1, int(Const.C_LIST_PADX * factor))
        scaled_pady: int = max(1, int(Const.C_LIST_PADY * factor))

        for _ in range(MAX_CARD_NUM):
            label: GUICardListLabel = GUICardListLabel(self.frame, self.udi_gui_frame, factor)
            label.pack(padx=scaled_padx, pady=scaled_pady)

            for child in label.children.values():
                child.bind("<MouseWheel>", self._on_mousewheel)

            label.bind("<MouseWheel>", self._on_mousewheel)
            self.card_list.append(label)
