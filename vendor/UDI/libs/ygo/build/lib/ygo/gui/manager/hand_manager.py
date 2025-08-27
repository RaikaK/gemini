#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import tkinter as tk
import PIL.ImageTk as Itk

from ygo import constants as c
from ygo import models as mdl

from .scollable_frame import ScrollableFrameX
from .util import extract_card_list_by_player_pos, generate_text_by_card_abstract, generate_card_overlay_text


class HandLabel(tk.Frame):
    def __init__(self, master, img, card:mdl.DuelCard, table_index, udi_gui_frame):
        tk.Frame.__init__(self, master)
        
        self.img = img
        self.card = card
        self.table_index = table_index
        self.udi_gui_frame  = udi_gui_frame

        img_label=tk.Label(self, image=self.img)
        img_label.pack()
        if self.table_index != -1:
            img_label.bind("<Button-1>", lambda event, _table_index = self.table_index:self.call_card_text_manager(_table_index))

    def call_card_text_manager(self, table_index):
        self.udi_gui_frame.card_text_manager.update_table_index(table_index)


class HandManager(ScrollableFrameX):
    def __init__(self, udi_gui_frame, player_id, master, **key):
        self.udi_gui_frame  = udi_gui_frame
        self.master=master
        self.player_id = player_id
        self.key = key

        super().__init__(master, **key)

    def reset(self):
        if self.once:
            self.frame.destroy()
            self.canvas.destroy()
            self.make_canvas_frame()
        self.once = True

    def update(self, duel_state_data:mdl.DuelStateData):
        self.reset()
        duel_card_table = duel_state_data.duel_card_table

        # self.player_idに対応するplayerのhandに含まれるカードを抽出
        card_list = extract_card_list_by_player_pos(duel_card_table, self.player_id, c.enums.PosId.HAND)
        for (table_index, card) in card_list:
            # カードリストの各カードから、情報を抽出
            overlay_text, _ = generate_text_by_card_abstract(duel_card_table, card)
            
            # 画像生成+更新部分
            img = self.udi_gui_frame.medium_image_manager.get_image_by_card(card)
            img = generate_card_overlay_text(img, overlay_text)
            tkimg = Itk.PhotoImage(img)
            label = HandLabel(self.frame, tkimg, card,table_index, self.udi_gui_frame)
            label.pack(side=tk.LEFT)

            for child in label.children.values():
                child.bind("<MouseWheel>", self._on_mousewheel)
            label.bind("<MouseWheel>", self._on_mousewheel)

