#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import tkinter as tk

from ygo.util.text import TextUtil
from ygo import models as mdl

from .const import Const



class DialogManager:
    def __init__(self, udi_gui_frame, master, **key):
        self.udi_gui_frame  = udi_gui_frame
        self.master=master
        self.key = key

        self.text_label=tk.Label(self.master, text="selection type, id", font=Const.DIALOG_FONT, wraplength=Const.DIALOG_WRAP_LENGTH)
        self.text_label.pack()


    def update(self, command_request:mdl.CommandRequest):
        text_util:TextUtil = self.udi_gui_frame.text_util

        # selection_type
        text = f"selection_type={command_request.selection_type}\n"
        # selectionId
        if command_request.selection_id >= 0:
            text += f"『{text_util.get_selection_id_text(command_request.selection_id)}』\n"

        self.text_label["text"] = text


