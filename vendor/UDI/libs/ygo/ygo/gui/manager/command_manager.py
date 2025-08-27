#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

from __future__ import annotations
import tkinter as tk
import PIL.ImageTk as Itk
import copy

from ygo.udi_io import UdiIO
from ygo import constants as c
from ygo import models as mdl

from .util import make_command_text, make_command_icon
from .const import Const
from .scollable_frame import ScrollableFrameY


class CommandLabel(tk.Frame):
    def __init__(self,command_manager:CommandManager, master, num, img, card:mdl.DuelCard, table_index, text,subtext, udi_gui_frame, ):
        tk.Frame.__init__(self, master)
        self.parent_command_manager = command_manager
        
        self.img = img
        self.card = card
        self.table_index = table_index
        self.text = text
        self.subtext = subtext
        self.ai_text = ""
        self.udi_gui_frame = udi_gui_frame
        self.num = num

        # コマンド番号
        num_label=tk.Label(self, text=str(num),font=Const.COMMAND_NUM_FONT)
        num_label.pack(side=tk.LEFT)

        # メインのカード
        img_label_dir = tk.Frame(self, width = Const.M_CARD_W, height=Const.M_CARD_H)
        img_label_dir.propagate(False)
        img_label_dir.pack(side=tk.LEFT)
        img_label=tk.Label(img_label_dir, image=self.img)
        img_label.pack(side=tk.TOP)
        if self.table_index != -1:
            img_label.bind("<Button-1>", lambda event, _table_index = self.table_index:self.call_card_text_manager(_table_index))

        # コマンドのテキスト
        text_dir = tk.Frame(self)
        text_dir.pack(side=tk.LEFT)
        self.text_label=tk.Label(text_dir, text=self.text, font=Const.COMMAND_TEXT_FONT, wraplength=Const.COMMAND_WRAP_LENGTH)
        self.text_label.pack(side=tk.TOP)
        self.subtext_label=tk.Label(text_dir, text=self.subtext, font=Const.COMMAND_SUBTEXT_FONT, wraplength=Const.COMMAND_WRAP_LENGTH)
        self.subtext_label.pack(side=tk.TOP)
        self.ai_text_label=tk.Label(text_dir, text=self.ai_text, font=Const.COMMAND_SUBTEXT_FONT, wraplength=Const.COMMAND_WRAP_LENGTH, foreground='#ff0000')
        self.ai_text_label.pack(side=tk.TOP)

        # コマンド実行ボタン
        self.b_send=tk.Button(self, text="実行", font=Const.COMMAND_BUTTON_FONT, command=self.send_command)
        self.b_send.pack(side=tk.LEFT,padx=Const.COMMAND_BUTTON_PADX, pady=Const.COMMAND_BUTTON_PADY) 
        
    def call_card_text_manager(self, table_index):
        self.udi_gui_frame.card_text_manager.update_table_index(table_index)

    def send_command(self):
        result = self.udi_gui_frame.set_queue(self.num)
        if result == True:
            self.parent_command_manager.disable_command_label()


class CommandManager(ScrollableFrameY):
    def __init__(self, udi_gui_frame, master, **key):
        self.udi_gui_frame  = udi_gui_frame
        self.master=master
        self.key = key

        self.label_list:list[CommandLabel]
        self.label_list = []
        
        super().__init__(master, **key)

    def reset(self):
        # CommandLableのリストをクリア
        self.label_list.clear()

        # キャンバス自体を削除
        if self.once:
            self.frame.destroy()
            self.canvas.destroy()
            self.make_canvas_frame()
        self.once = True

    def disable_command_label(self):
        for label in self.label_list:
            label.b_send.config(state=tk.DISABLED)

    def highlight_selected_command(self,command_num):
        self.label_list[command_num].config(highlightthickness=3, highlightbackground = Const.HIGHLIGHT_BLACK, highlightcolor = Const.HIGHLIGHT_BLACK)

    def highlight_ai_command(self,command_num):
        self.label_list[command_num].config(background = Const.HIGHLIGHT_GRAY)

    def display_ai_info(self,ai_info_list):
        for i, ai_info in enumerate(ai_info_list):
            self.label_list[i].ai_text_label["text"] = ai_info
            

    def update(self, command_request:mdl.CommandRequest, duel_state_data:mdl.DuelStateData):
        self.reset()

        duel_card_table = duel_state_data.duel_card_table
        commands = command_request.commands
        # 各コマンドから情報を取得
        for i, command in enumerate(commands):
            ################################################################################################################
            # コマンドに関係するカード
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
            icon_type, icon_id = make_command_icon(command)
            
            if icon_type == UdiIO.RatingTextType.CARD_ID or icon_type == UdiIO.RatingTextType.ETC:
                if table_index == -1:
                    # ダミーのカードを作成
                    dummy_dict = {"cardId" : 0, "playerId" : 0, "posId" : -1, "cardIndex" : -1, "face" : 0, "turn" : 0, "isDisabled" : -1, 
                              "atkVal" : -1, "defVal" : -1, "isAttacking" : -1, "isAttacked" : -1, "equipTarget" : -1, "magicCounterNum" : -1, 
                              "usedEffect1" : -1, "usedEffect2" : -1, "usedEffect3" : -1, "turnPassed" : -1, "level" : -1}
                    card = mdl.DuelCard(dummy_dict)
                # get_image_by_cardでcardの画像を取得するが，セットコマンドのときは横向きにする(厳密には魔法罠のセットは横向きではないが，視認性のため妥協)
                if command.command_type == c.CommandType.SET or command.command_type == c.CommandType.SET_MONST:
                    card = copy.deepcopy(card)
                    card.turn = 1
                img = self.udi_gui_frame.medium_image_manager.get_image_by_card(card)
            else:
                icon_id = int(icon_id)
                img = self.udi_gui_frame.medium_image_manager.get_icon_image(icon_type, icon_id)

            tkimg = Itk.PhotoImage(img) # カード、アイコンの画像

            label = CommandLabel(self, self.frame, i, tkimg, card, table_index, text, subtext, self.udi_gui_frame,)
            label.pack(pady=Const.COMMAND_PADY, side=tk.TOP, anchor=tk.W)
            self.label_list.append(label)

            for child in label.children.values():
                child.bind("<MouseWheel>", self._on_mousewheel)
                for g_child in child.children.values():
                    g_child.bind("<MouseWheel>", self._on_mousewheel)
            label.bind("<MouseWheel>", self._on_mousewheel)
            