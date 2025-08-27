#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import copy
import tkinter as tk

import PIL.ImageTk as Itk

from ygo import constants as c
from ygo import models as mdl

from .const import Const
from .scollable_frame import ScrollableFrameY
from .util import extract_card_list_by_player_pos, generate_text_by_card_abstract, generate_card_overlay_text


class CardLabel(tk.Frame):
    def __init__(self, master, udi_gui_frame):
        tk.Frame.__init__(self, master, relief=tk.SOLID, bd=Const.C_LIST_BD)

        self.udi_gui_frame  = udi_gui_frame

        self.img = None
        self.text = None
        self.card = None
        self.table_index = None

        self.img_label=tk.Label(self, image="")
        self.img_label.pack(side = tk.TOP)

        self.info_label=tk.Label(self, text = "", font=Const.C_LIST_FONT, wraplength=Const.C_LIST_WRAP_LENGTH)
        self.info_label.pack(side = tk.TOP)

        self.func_id1 = None
        self.func_id2 = None

    def update(self, tkimg, side_text, card, table_index):
        self.img = tkimg
        self.text = side_text
        self.card = copy.deepcopy(card)
        self.table_index = table_index

        self.img_label.config(image=self.img)
        self.img_label.image = self.img
        self.info_label.config(text=side_text)

        if self.table_index != -1:
            self.func_id1 = self.img_label.bind("<Button-1>", lambda event, _table_index = self.table_index:self.call_card_text_manager(_table_index))
            self.func_id2 = self.info_label.bind("<Button-1>", lambda event, _table_index = self.table_index:self.call_card_text_manager(_table_index))
        else:
            if self.func_id1 is not None:
                self.img_label.unbind("<Button-1>",self.func_id1)
                self.func_id1 = None
            if self.func_id2 is not None:
                self.info_label.unbind("<Button-1>", self.func_id2)
                self.func_id2 = None

    def reset(self):
        if self.func_id1 is not None:
            self.img_label.unbind("<Button-1>",self.func_id1)
            self.func_id1 = None
        if self.func_id2 is not None:
            self.info_label.unbind("<Button-1>", self.func_id2)
            self.func_id2 = None

        self.img = None
        self.text = None
        self.card = None
        self.table_index = None

        self.img_label.config(image="")
        self.info_label.config(text="")


    def call_card_text_manager(self, table_index):
        self.udi_gui_frame.card_text_manager.update_table_index(table_index)


class CardListManager(ScrollableFrameY):
    def __init__(self, udi_gui_frame, master, **key):
        self.udi_gui_frame  = udi_gui_frame
        self.master = master
        self.key = key

        self.player_id = c.enums.PlayerId.NO_VALUE
        self.pos_id    = c.enums.PosId.NO_VALUE
        self.duel_card_table = None

        self.info_label = tk.Label(self.master, text= f"P{self.player_id}:{c.enums.PosId(self.pos_id)}", font=Const.C_LIST_INFO_FONT, wraplength=Const.C_LIST_INFO_WRAP_LENGTH)
        self.info_label.pack()

        super().__init__(master, **key)

        self.duel_card_table = None

        self.card_list = []
        MAX_CARD_NUM = 60+15 # リストに入りうるカードの最大枚数
        for _ in range(MAX_CARD_NUM):
            label = CardLabel(self.frame, self.udi_gui_frame)
            label.pack(padx=Const.C_LIST_PADX,pady=Const.C_LIST_PADY)
            for child in label.children.values():
                child.bind("<MouseWheel>", self._on_mousewheel)
            label.bind("<MouseWheel>", self._on_mousewheel)
            self.card_list.append(label)

    def set_player_pos(self, player_id, pos_id):
        # 不要な更新を避けるために，player_idとpos_idが変わったときだけ更新する
        # if self.player_id != player_id or self.pos_id != pos_id:
        # →リストボタンを押したときは常にリストを更新するように変更
        self.player_id = player_id
        self.pos_id  = pos_id
        self.reset()
        self.update()


    # board上のカードをクリックしたときに，card_tableからcardを取得してplayer_idとpos_idを更新する
    def set_player_pos_by_table_index(self, table_index):
        card = self.duel_card_table[table_index]
        self.set_player_pos(card.player_id, card.pos_id)


    def set_duel_card_table(self, duel_state_data:mdl.DuelStateData):
        duel_card_table = duel_state_data.duel_card_table
        self.duel_card_table = duel_card_table
        # duel_card_tableが更新されたときに，リストを更新する
        # self.update()
        # →重いので、代わりにresetだけしておく
        # なお，毎回updateだけだと，カードリストに矛盾が生じるという問題もある
        self.reset()

    def reset(self):
        for card in self.card_list:
            card.reset()

    def update(self):
        # カードリストがどのplayer_idのpos_idを対象にしているのかをテキストで表示
        self.info_label.config(text=f"{c.enums.PlayerId(self.player_id)}:{c.enums.PosId(self.pos_id)}")

        if self.duel_card_table is not None:
            # 対象player_idのpos_idに含まれるカードを抽出
            card_list = extract_card_list_by_player_pos(self.duel_card_table, self.player_id, self.pos_id)
            for i, (table_index, card) in enumerate(card_list):
                # カードリストの各カードから、情報を抽出
                overlay_text, side_text = generate_text_by_card_abstract(self.duel_card_table, card)

                # 画像生成+更新部分
                img = self.udi_gui_frame.medium_image_manager.get_image_by_card(card)
                img = generate_card_overlay_text(img, overlay_text)
                tkimg = Itk.PhotoImage(img)
                self.card_list[i].update(tkimg, side_text, card, table_index)



