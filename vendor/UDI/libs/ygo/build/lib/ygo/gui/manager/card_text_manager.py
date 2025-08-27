#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import tkinter as tk
import PIL.ImageTk as Itk

from ygo import constants as c
from ygo import models as mdl

from .const import Const
from .scollable_frame import ScrollableFrameY
from .util import generate_text_by_card_detailed, generate_card_info_text


class CardTextLabel(tk.Frame):
    def __init__(self, master, img):
        tk.Frame.__init__(self, master)
        
        self.default_img = img
        self.img = img
        self.default_text = ""
        self.text = ""
        self.default_name = ""
        self.name = ""

        self.img_label=tk.Label(self, image=self.img)
        self.img_label.pack(side=tk.LEFT)
        
        self.name_label=tk.Label(self, text=self.name, font=Const.CARDTEXT_NAME_FONT, wraplength=Const.CARDTEXT_WRAP_LENGTH)
        self.name_label.pack(side=tk.TOP)

        self.text_label=tk.Label(self, text=self.text, font=Const.CARDTEXT_FONT, wraplength=Const.CARDTEXT_WRAP_LENGTH)
        self.text_label.pack(side=tk.TOP)
        
    def reset(self):
        self.img = None
        self.text = None
        self.name = None

        self.img_label.config(image=self.default_img)
        self.img_label.image = self.default_img
        self.name_label.config(text=self.default_name)
        self.text_label.config(text=self.default_text)
    
    def update(self, img, text, name):
        self.img = img
        self.text = text
        self.name = name

        self.img_label.config(image=self.img)
        self.img_label.image = self.img
        self.text_label.config(text=text)
        self.name_label.config(text=name)


class CardTextManager(ScrollableFrameY):
    def __init__(self, udi_gui_frame, master, **key):
        self.udi_gui_frame  = udi_gui_frame
        self.master=master
        self.key = key

        self.duel_card_table = None
        self.table_index = None

        super().__init__(master, **key)

        # CardTextlable配置
        img = self.udi_gui_frame.large_image_manager.get_protector_image()
        tkimg = Itk.PhotoImage(img)
        text = ""
        self.label = CardTextLabel(self.frame, tkimg)
        self.label.pack()

        for child in self.label.children.values():
            child.bind("<MouseWheel>", self._on_mousewheel)
        self.label.bind("<MouseWheel>", self._on_mousewheel)

    def set_duel_card_table(self, duel_state_data : mdl.DuelStateData):
        duel_card_table = duel_state_data.duel_card_table
        self.duel_card_table = duel_card_table
        if self.table_index is not None:
            self.update_table_index(self.table_index)

    def reset(self):
        self.label.reset()

    # duel_card_tableのindexで詳細表示を更新
    def update_table_index(self, table_index):
        self.reset()
        self.table_index = table_index

        card:mdl.DuelCard = self.duel_card_table[self.table_index]
        card_id = card.card_id
        img = self.udi_gui_frame.large_image_manager.get_card_image(card_id)
        tkimg = Itk.PhotoImage(img)
        
        # カードテキスト取得
        text = ""
        try:
            card_name = self.udi_gui_frame.card_util.get_name(card_id)
            card_text = self.udi_gui_frame.card_util.get_text(card_id)
        except KeyError:
            card_name = "(不明)"
            card_text = ""
        text += card_text
        text += "\n\n"
        
        card_info_text = generate_card_info_text(card_id, self.udi_gui_frame.card_util)
        text += card_info_text
        text += "\n\n"

        # table indexも表示
        text += "--------------------------------------------------\n"
        text += f"table index: {self.table_index}\n\n"

        # 詳細情報追加
        detailed_text = generate_text_by_card_detailed(self.duel_card_table, card, self.udi_gui_frame.card_util)
        text += "info from table:\n"
        text += detailed_text
        
        text += "\ninfo from table(raw):\n"
        text += str(card)

        # 表示更新
        self.label.update(tkimg, text, card_name)
