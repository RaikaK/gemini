#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import tkinter as tk

from ygo.util.text import TextUtil

from .const import Const
from .scollable_frame import ScrollableFrameY


class LogLabel(tk.Frame):
    def __init__(self, master, num, text, udi_gui_frame):
        tk.Frame.__init__(self, master, relief=tk.SOLID, bd=Const.LOG_BD)
        
        self.text = text
        self.udi_gui_frame  = udi_gui_frame

        num_label=tk.Label(self, text=str(num), font=Const.LOG_NUM_FONT)
        num_label.pack()

        text_label=tk.Label(self, text=self.text, font=Const.LOG_TEXT_FONT, wraplength=Const.LOG_WRAP_LENGTH)
        text_label.pack()



class LogManager(ScrollableFrameY):
    def __init__(self, udi_gui_frame, master, **key):
        self.udi_gui_frame=udi_gui_frame
        self.master=master
        self.key=key
        self.label_list = []
        super().__init__(master, **key)

    def reset(self):
        self.label_list.clear() # これを追加して，デュエル終了時にresetを呼ばないと，前の試合のデータが残る
        self.frame.destroy()
        self.canvas.destroy()
        self.make_canvas_frame()

    def update(self, udi_io_duel_log):
        # self.reset()
        text_util:TextUtil = self.udi_gui_frame.text_util

        # ログが増えた場合は追加
        if len(self.label_list) < len(udi_io_duel_log):
            for i in range(len(self.label_list), len(udi_io_duel_log)):
                log = udi_io_duel_log[i]
                text = text_util.get_duel_log_entry_text(log)
                label = LogLabel(self.frame, i, text, self.udi_gui_frame)
                label.pack(pady=Const.LOG_PADY)
                for child in label.children.values():
                    child.bind("<MouseWheel>", self._on_mousewheel)
                label.bind("<MouseWheel>", self._on_mousewheel)
                self.label_list.append(label)
        # ログが減った場合は削除
        elif len(self.label_list) > len(udi_io_duel_log):
            for i in range(len(udi_io_duel_log), len(self.label_list)):
                self.label_list[i].destroy()
            self.label_list = self.label_list[:len(udi_io_duel_log)]

        # for i, log in enumerate(udi_io_duel_log):
        #     text = text_util.get_duel_log_entry_text(log)
        #     label = LogLabel(self.frame, i, text, self.udi_gui_frame)
        #     label.pack(pady=Const.LOG_PADY)
        #     for child in label.children.values():
        #         child.bind("<MouseWheel>", self._on_mousewheel)
        #     label.bind("<MouseWheel>", self._on_mousewheel)
        
        # # 最初に追加したログにスクロール
        # self.canvas.yview_moveto(0.0)

    def highlight_log_diff(self, before_log_length, before_command_log_length):
        # まず全てのログをデフォルト色に戻す(いったん変更されているところまで逆順にアクセスして，その後はデフォルトのままだったらbreakする)
        # 前回コマンドを要求されたタイミングでのログの長さが保存されている場合は，highlightthicknessが変更されているところを目指して逆順にアクセス
        if before_command_log_length is not None:
            _flag = False
            for i in range(len(self.label_list)-1, -1, -1):
                if self.label_list[i].cget("highlightthickness") == 0:
                    if _flag:
                        break
                else:
                    _flag = True
                    self.label_list[i].config(highlightthickness=0, highlightbackground ="SystemButtonFace", highlightcolor = "SystemWindowFrame")
                    self.label_list[i].config(bg=Const.DEFAULT_COLOR)
        else:
            # 保存されていない場合は，背景色が変更されているところを目指す
            _flag = False
            for i in range(len(self.label_list)-1, -1, -1):
                if self.label_list[i].cget("bg") == Const.DEFAULT_COLOR:
                    if _flag:
                        break
                else:
                    _flag = True
                    self.label_list[i].config(bg=Const.DEFAULT_COLOR)
        
        # 変更があったログをハイライト
        for i in range(before_log_length, len(self.label_list)):
            self.label_list[i].config(bg=Const.HIGHLIGHT_GRAY)

        # 前回コマンドを要求されたタイミングでのログの長さが保存されている場合，そのログをハイライト
        if before_command_log_length is not None:
            for i in range(before_command_log_length, len(self.label_list)):
                self.label_list[i].config(highlightthickness=2, highlightbackground = Const.HIGHLIGHT_BLACK, highlightcolor = Const.HIGHLIGHT_BLACK)
