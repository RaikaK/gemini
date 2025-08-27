#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import tkinter as tk
import PIL.ImageTk as Itk

from ygo import constants as c
from ygo import models as mdl

from .util import make_command_text
from .const import Const
from .scollable_frame import ScrollableFrameY


class CommandLabel(tk.Frame):
    def __init__(self, master, img, card:mdl.DuelCard, table_index, text, subtext, udi_gui_frame):
        tk.Frame.__init__(self, master)
        
        self.img = img
        self.card = card
        self.table_index = table_index
        self.text = text
        self.subtext = subtext
        self.udi_gui_frame = udi_gui_frame

        # メインのカード
        img_label=tk.Label(self, image=self.img)
        img_label.pack(side=tk.LEFT)
        if self.table_index != -1:
            img_label.bind("<Button-1>", lambda event, _table_index = self.table_index: self.call_card_text_manager(_table_index))

        # コマンドのテキスト
        text_dir = tk.Frame(self)
        text_dir.pack(side=tk.LEFT)
        self.text_label=tk.Label(text_dir, text=self.text, font=Const.CONTEXT_TEXT_FONT, wraplength=Const.CONTEXT_WRAP_LENGTH)
        self.text_label.pack(side=tk.TOP)
        self.subtext_label=tk.Label(text_dir, text=self.subtext, font=Const.CONTEXT_SUBTEXT_FONT, wraplength=Const.CONTEXT_WRAP_LENGTH)
        self.subtext_label.pack(side=tk.TOP)

    def call_card_text_manager(self, table_index):
        self.udi_gui_frame.card_text_manager.update_table_index(table_index)

    # TODO: もしかしたら必要かも
    def update_wraplength(self, event):
        self.text_label.config(wraplength=self.winfo_width())


class ContextManager(ScrollableFrameY):
    def __init__(self, udi_gui_frame, master, **key):
        self.udi_gui_frame  = udi_gui_frame
        self.master=master
        self.key = key

        super().__init__(master, **key)

    def reset(self):
        if self.once:
            self.frame.destroy()
            self.canvas.destroy()
            self.make_canvas_frame()

        self.once = True

    def update(self, command_request:mdl.CommandRequest, duel_state_data:mdl.DuelStateData):
        self.reset()

        duel_card_table = duel_state_data.duel_card_table
        command_log = command_request.command_log
        for log in command_log:
            ################################################################################################################
            # コマンドに関係するカード
            command = log.command
            table_index = command.table_index
            card : mdl.DuelCard
            if table_index == -1:
                card = None
                card_id = -1
            else:
                card = duel_card_table[table_index]
                card_id = card.card_id

            # コマンドをテキスト化
            text = ""
            if table_index == -1:
                pass
            else:
                if table_index < 100:
                    text += "(自分)"
                else:
                    text += "(相手)"

                if card_id in (0, -1):
                    text += "裏側カード" # TODO:table_indexは指定されているのにcard_id==0ということは、何のカードか不明ということ?
                else:
                    try:
                        text += self.udi_gui_frame.card_util.get_name(card_id)
                    except KeyError:
                        text += "不明カード"
            text += "\n"
            text += make_command_text(command)
            
            # コマンドをそのまま表示
            subtext = str(command)

            ################################################################################################################
            # 画像生成+GUI反映部分
            if table_index == -1:
                # ダミーのカードを作成
                dummy_dict = {"cardId" : 0, "playerId" : 0, "posId" : -1, "cardIndex" : -1, "face" : 0, "turn" : 0, "isDisabled" : -1, 
                              "atkVal" : -1, "defVal" : -1, "isAttacking" : -1, "isAttacked" : -1, "equipTarget" : -1, "magicCounterNum" : -1, 
                              "usedEffect1" : -1, "usedEffect2" : -1, "usedEffect3" : -1, "turnPassed" : -1, "level" : -1}
                card = mdl.DuelCard(dummy_dict)
            img = self.udi_gui_frame.small_image_manager.get_image_by_card(card)
            tkimg = Itk.PhotoImage(img)

            label = CommandLabel(self.frame, tkimg, card, table_index, text, subtext, self.udi_gui_frame)
            label.pack(side=tk.TOP, anchor=tk.W)

            for child in label.children.values():
                child.bind("<MouseWheel>", self._on_mousewheel)
                for g_child in child.children.values():
                    g_child.bind("<MouseWheel>", self._on_mousewheel)
            label.bind("<MouseWheel>", self._on_mousewheel)