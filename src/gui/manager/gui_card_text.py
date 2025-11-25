import PIL.ImageTk as Itk
import tkinter as tk

from ygo import models as mdl
from ygo.gui.manager.card_text_manager import CardTextLabel, CardTextManager
from ygo.gui.manager.const import Const
from ygo.gui.manager.scollable_frame import ScrollableFrameY
from ygo.gui.manager.util import generate_text_by_card_detailed, generate_card_info_text


class GUICardTextLabel(CardTextLabel):
    """
    GUIカードテキストラベル
    """

    def __init__(self, master: tk.Misc, img: Itk.PhotoImage, factor: float) -> None:
        """
        初期化する。
        """
        super().__init__(master, img)

        scaled_name_font: tuple = (
            Const.CARDTEXT_NAME_FONT[0],
            max(8, int(int(Const.CARDTEXT_NAME_FONT[1]) * factor)),
            Const.CARDTEXT_NAME_FONT[2],
        )
        scaled_text_font: tuple = (
            Const.CARDTEXT_FONT[0],
            max(8, int(int(Const.CARDTEXT_FONT[1]) * factor)),
        )
        scaled_wrap_length: int = int(Const.CARDTEXT_WRAP_LENGTH * factor)

        self.name_label.config(font=scaled_name_font, wraplength=scaled_wrap_length)
        self.text_label.config(font=scaled_text_font, wraplength=scaled_wrap_length)


class GUICardText(CardTextManager):
    """
    GUIカードテキスト
    """

    def __init__(self, udi_gui_frame, master: tk.Misc, **key) -> None:
        """
        初期化する。
        """
        factor: float = udi_gui_frame.factor

        self.udi_gui_frame = udi_gui_frame
        self.master: tk.Misc = master
        self.key: dict = key

        self.duel_card_table: list[mdl.DuelCard] | None = None
        self.table_index: int | None = None

        ScrollableFrameY.__init__(self, master, **key)

        img = self.udi_gui_frame.large_image_manager.get_protector_image()
        tkimg: Itk.PhotoImage = Itk.PhotoImage(img)
        self.label: GUICardTextLabel = GUICardTextLabel(self.frame, tkimg, factor)
        self.label.pack()

        for child in self.label.children.values():
            child.bind("<MouseWheel>", self._on_mousewheel)

        self.label.bind("<MouseWheel>", self._on_mousewheel)

    def update_table_index(self, table_index: int) -> None:
        """
        更新する。
        """
        self.reset()
        self.table_index = table_index

        card: mdl.DuelCard = self.duel_card_table[self.table_index]
        card_id: int = card.card_id
        img = self.udi_gui_frame.large_image_manager.get_card_image(card_id)
        tkimg: Itk.PhotoImage = Itk.PhotoImage(img)

        # カードテキスト取得
        text: str = ""

        try:
            card_name: str = self.udi_gui_frame.card_util.get_name(card_id)
            card_text: str = self.udi_gui_frame.card_util.get_text(card_id)
        except KeyError:
            card_name = "(不明)"
            card_text = ""

        text += card_text

        text += "\n\n"
        card_info_text: str = generate_card_info_text(card_id, self.udi_gui_frame.card_util)
        text += card_info_text

        if self.udi_gui_frame.debug_mode:
            # table indexも表示
            text += "\n\n"
            text += "--------------------------------------------------\n"
            text += f"table index: {self.table_index}\n\n"

            # 詳細情報追加
            detailed_text: str = generate_text_by_card_detailed(
                self.duel_card_table, card, self.udi_gui_frame.card_util
            )
            text += "info from table:\n"
            text += detailed_text

            text += "\ninfo from table(raw):\n"
            text += str(card)

        # 表示更新
        self.label.update(tkimg, text, card_name)
