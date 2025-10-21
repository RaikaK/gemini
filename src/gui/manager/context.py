#!/usr/bin/env python
import tkinter as tk
import PIL.ImageTk as Itk

from ygo import models as mdl
from ygo.gui.manager.util import make_command_text
from ygo.gui.manager.const import Const
from ygo.gui.manager.context_manager import CommandLabel, ContextManager
from ygo.models.command_request import CommandEntry, CommandLogEntry


class GUIContextLabel(CommandLabel):
    """
    GUIコマンドラベル
    """

    def __init__(
        self,
        master: tk.Misc,
        img: Itk.PhotoImage,
        card: mdl.DuelCard,
        table_index: int,
        text: str,
        subtext: str,
        udi_gui_frame,
        factor: float,
    ) -> None:
        """
        初期化する
        """
        tk.Frame.__init__(self, master)

        self.img: Itk.PhotoImage = img
        self.card: mdl.DuelCard = card
        self.table_index: int = table_index
        self.text: str = text
        self.subtext: str = subtext
        self.udi_gui_frame = udi_gui_frame

        scaled_text_font: tuple = (
            Const.CONTEXT_TEXT_FONT[0],
            max(7, int(int(Const.CONTEXT_TEXT_FONT[1]) * factor)),
        )
        scaled_subtext_font: tuple = (
            Const.CONTEXT_SUBTEXT_FONT[0],
            max(7, int(int(Const.CONTEXT_SUBTEXT_FONT[1]) * factor)),
        )
        scaled_wrap_length: int = int(Const.CONTEXT_WRAP_LENGTH * factor)

        # メインのカード
        img_label: tk.Label = tk.Label(self, image=self.img)
        img_label.pack(side=tk.LEFT)
        if self.table_index != -1:
            img_label.bind(
                "<Button-1>", lambda event, _table_index=self.table_index: self.call_card_text_manager(_table_index)
            )

        # コマンドのテキスト
        text_dir: tk.Frame = tk.Frame(self)
        text_dir.pack(side=tk.LEFT)
        self.text_label: tk.Label = tk.Label(
            text_dir, text=self.text, font=scaled_text_font, wraplength=scaled_wrap_length
        )
        self.text_label.pack(side=tk.TOP)
        self.subtext_label: tk.Label = tk.Label(
            text_dir, text=self.subtext, font=scaled_subtext_font, wraplength=scaled_wrap_length
        )
        self.subtext_label.pack(side=tk.TOP)


class GUIContext(ContextManager):
    """
    GUIコンテキスト
    """

    def update(self, command_request: mdl.CommandRequest, duel_state_data: mdl.DuelStateData) -> None:
        """
        更新する。
        """
        self.reset()

        factor: float = self.udi_gui_frame.factor

        duel_card_table: list[mdl.DuelCard] = duel_state_data.duel_card_table
        command_log: list[CommandLogEntry] = command_request.command_log

        for log in command_log:
            ##################################################
            # コマンドに関係するカード
            command: CommandEntry = log.command
            table_index: int = command.table_index
            card: mdl.DuelCard
            card_id: int

            if table_index == -1:
                card = None
                card_id = -1
            else:
                card = duel_card_table[table_index]
                card_id = card.card_id

            # コマンドをテキスト化
            text: str = ""

            if table_index == -1:
                pass
            else:
                if table_index < 100:
                    text += "(自分)"
                else:
                    text += "(相手)"

                if card_id in (0, -1):
                    text += "裏側カード"
                else:
                    try:
                        text += self.udi_gui_frame.card_util.get_name(card_id)
                    except KeyError:
                        text += "不明カード"
            text += "\n"
            text += make_command_text(command)

            # コマンドをそのまま表示
            subtext: str = str(command)

            ##################################################
            # 画像生成+GUI反映部分
            if table_index == -1:
                dummy_dict: dict = {
                    "cardId": 0,
                    "playerId": 0,
                    "posId": -1,
                    "cardIndex": -1,
                    "face": 0,
                    "turn": 0,
                    "isDisabled": -1,
                    "atkVal": -1,
                    "defVal": -1,
                    "isAttacking": -1,
                    "isAttacked": -1,
                    "equipTarget": -1,
                    "magicCounterNum": -1,
                    "usedEffect1": -1,
                    "usedEffect2": -1,
                    "usedEffect3": -1,
                    "turnPassed": -1,
                    "level": -1,
                }
                card = mdl.DuelCard(dummy_dict)

            img = self.udi_gui_frame.small_image_manager.get_image_by_card(card)
            tkimg = Itk.PhotoImage(img)

            label: GUIContextLabel = GUIContextLabel(
                self.frame, tkimg, card, table_index, text, subtext, self.udi_gui_frame, factor
            )
            label.pack(side=tk.TOP, anchor=tk.W)

            for child in label.children.values():
                child.bind("<MouseWheel>", self._on_mousewheel)

                for g_child in child.children.values():
                    g_child.bind("<MouseWheel>", self._on_mousewheel)

            label.bind("<MouseWheel>", self._on_mousewheel)
