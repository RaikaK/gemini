#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import tkinter as tk
import PIL.ImageTk as Itk

from ygo import constants as c
from ygo import models as mdl

from .const import Const
from .scollable_frame import ScrollableFrameY


class ChainLabel(tk.Frame):
    def __init__(self, master, num, img, text, card:mdl.DuelCard, table_index, udi_gui_frame):
        tk.Frame.__init__(self, master,  relief=tk.SOLID, bd=Const.CHAIN_BD)
        
        self.udi_gui_frame  = udi_gui_frame

        self.img = img
        self.text = text
        self.card = card
        self.table_index = table_index

        self.num_label=tk.Label(self, text=str(num), font=Const.CHAIN_NUM_FONT)
        self.num_label.pack(side=tk.LEFT)
        if self.table_index != -1:
            self.num_label.bind("<Button-1>", lambda event, _table_index = self.table_index:self.call_card_text_manager(_table_index))

        self.img_label=tk.Label(self, image=self.img)
        self.img_label.pack(side=tk.TOP)
        if self.table_index != -1:
            self.img_label.bind("<Button-1>", lambda event, _table_index = self.table_index:self.call_card_text_manager(_table_index))

        self.text_label=tk.Label(self, text=self.text, font=Const.CHAIN_TEXT_FONT, wraplength=Const.CHAIN_WRAP_LENGTH)
        self.text_label.pack(side=tk.TOP)
        if self.table_index != -1:
            self.text_label.bind("<Button-1>", lambda event, _table_index = self.table_index:self.call_card_text_manager(_table_index))

    def call_card_text_manager(self, table_index):
        self.udi_gui_frame.card_text_manager.update_table_index(table_index)


class ChainManager(ScrollableFrameY):
    def __init__(self, udi_gui_frame, master, **key):
        self.udi_gui_frame = udi_gui_frame
        self.master=master
        self.key = key

        super().__init__(master, **key)


    def reset(self):
        if self.once:
            self.frame.destroy()
            self.canvas.destroy()
            self.make_canvas_frame()
        self.once = True


    def update(self, duel_state_data : mdl.DuelStateData):
        self.reset()

        # 各チェーンから情報を取得
        duel_card_table = duel_state_data.duel_card_table
        chain_stack = duel_state_data.chain_stack
        for i, chain in enumerate(chain_stack):
            text = ""

            # 効果を発動したカード
            table_index = chain.table_index
            card = duel_card_table[table_index]
            if table_index  < 100:
                text+="(自分)"
            else:
                text+="(相手)"

            # 効果番号
            effect_no = chain.effect_no
            text += f'{c.enums.EffectNo(effect_no)}'
            text += "\n"

            # チェーンの状態
            chain_state = chain.chain_state
            text += f'{c.enums.ChainState(chain_state)}'
            text += "\n"

            # 効果の対象
            target_table_index_list = chain.target_table_index_list
            if len(target_table_index_list) > 0:
                text += "対象："
            for target_table_index in target_table_index_list:
                if target_table_index < 100:
                    text += "(自分)"
                else:
                    text += "(相手)"

                target_card:mdl.DuelCard = duel_card_table[target_table_index]
                target_card_id = target_card.card_id
                if target_card_id in (0, -1):
                    text += "裏側カード "
                else:
                    try:
                        text += self.udi_gui_frame.card_util.get_name(target_card_id)
                    except KeyError:
                        text += "不明 "
                    text += " "

            ################################################################################################################
            # 画像生成+GUI反映部分
            img = self.udi_gui_frame.medium_image_manager.get_image_by_card(card)
            tkimg = Itk.PhotoImage(img)

            label = ChainLabel(self.frame, i, tkimg, text, card, table_index, self.udi_gui_frame)
            label.pack()

            for child in label.children.values():
                child.bind("<MouseWheel>", self._on_mousewheel)
            label.bind("<MouseWheel>", self._on_mousewheel)

