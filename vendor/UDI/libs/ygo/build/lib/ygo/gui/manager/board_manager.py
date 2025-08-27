#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import copy
import tkinter as tk

from PIL import ImageDraw
import PIL.Image as I
import PIL.ImageTk as Itk

from ygo.udi_io import UdiIO
from ygo import constants as c
from ygo import models as mdl

from .const import Const
from .util import generate_text_by_card_abstract, generate_card_overlay_text,extract_card_list_by_player_pos


class CardLabel(tk.Frame):
    def __init__(self, master, bg_img:Itk.PhotoImage, udi_gui_frame):
        tk.Frame.__init__(self, master)

        self.bg_img = bg_img
        self.udi_gui_frame  = udi_gui_frame

        self.card = None
        self.table_index = None
        self.img = None

        self.img_label=tk.Label(self, image=self.bg_img, relief="ridge", width=self.bg_img.width(), height=self.bg_img.height(), bg="gray")
        self.img_label.pack()

        self.func_id = None

    def update(self, tkimg, card, table_index):
        self.img = tkimg
        self.card = copy.deepcopy(card)
        self.table_index = table_index
        
        self.img_label.config(image=self.img)
        self.img_label.image = self.img
        
        if self.table_index is None:
            pass
        if self.table_index != -1:
            self.func_id = self.img_label.bind("<Button-1>", lambda event, _table_index = self.table_index:self.call_card_text_manager(_table_index))
        else:
            if self.func_id is not None:
                self.img_label.unbind("<Button-1>",self.func_id)
                self.func_id = None

    def reset(self):
        if self.func_id is not None:
            self.img_label.unbind("<Button-1>",self.func_id)
            self.func_id = None

        self.img = None
        self.card = None
        self.table_index = None

        self.img_label.config(image=self.bg_img)
        self.img_label.image = self.bg_img


    def call_card_text_manager(self, table_index):
        self.udi_gui_frame.card_text_manager.update_table_index(table_index)

        # 追加してみた
        self.udi_gui_frame.card_list_manager.set_player_pos_by_table_index(table_index)


class PositionLabel(tk.Frame):
    def __init__(self, master, img:Itk.PhotoImage, player_id, pos_id, udi_gui_frame):
        tk.Frame.__init__(self, master)
        self.img = img
        self.pos_id = pos_id
        self.player_id = player_id
        self.udi_gui_frame  = udi_gui_frame

        self.text_label=tk.Button(self, text=f"{c.enums.PosId(self.pos_id)}\n", font=Const.BOARD_POSITION_FONT, relief="raised",bg="lightgray",command=self.call_card_list_manager)
        self.text_label.pack(pady=2) 

    def update(self, num):
        text=f"{c.enums.PosId(self.pos_id)}" + "\n" + str(num) + "枚"
        self.text_label.config(text=text)

    def call_card_list_manager(self):
        self.udi_gui_frame.card_list_manager.set_player_pos(self.player_id, self.pos_id)


class BoardManager:
    def __init__(self, udi_gui_frame, master, **key):
        self.udi_gui_frame  = udi_gui_frame
        self.master=master
        self.key = key

        # 各マスの背景画像
        card_width, card_height = Const.S_CARD_W, Const.S_CARD_H
        bg_img = I.new("RGB", (card_width, card_height), "gray")
        draw = ImageDraw.Draw(bg_img)
        bg_tkimg = Itk.PhotoImage(bg_img)

        # カード関連
        self.board = {c.enums.PlayerId.MYSELF : dict(), c.enums.PlayerId.RIVAL : dict()}
        row_col    = {c.enums.PlayerId.MYSELF : {c.enums.PosId.MONSTER_L_L :(5,2), c.enums.PosId.MONSTER_L   :(5,3), c.enums.PosId.MONSTER_C:(5,4), c.enums.PosId.MONSTER_R:(5,5), c.enums.PosId.MONSTER_R_R:(5,6),
                                                 c.enums.PosId.EX_L_MONSTER:(4,3), c.enums.PosId.EX_R_MONSTER:(4,5),
                                                 c.enums.PosId.MAGIC_L_L   :(6,2), c.enums.PosId.MAGIC_L     :(6,3), c.enums.PosId.MAGIC_C  :(6,4), c.enums.PosId.MAGIC_R  :(6,5), c.enums.PosId.MAGIC_R_R  :(6,6),
                                                 c.enums.PosId.FIELD       :(5,1)
                                                 },
                      c.enums.PlayerId.RIVAL  : {c.enums.PosId.MONSTER_L_L :(3,6), c.enums.PosId.MONSTER_L   :(3,5), c.enums.PosId.MONSTER_C:(3,4), c.enums.PosId.MONSTER_R:(3,3), c.enums.PosId.MONSTER_R_R:(3,2),
                                                 c.enums.PosId.EX_L_MONSTER:(4,5), c.enums.PosId.EX_R_MONSTER:(4,3),
                                                 c.enums.PosId.MAGIC_L_L   :(2,6), c.enums.PosId.MAGIC_L     :(2,5), c.enums.PosId.MAGIC_C  :(2,4), c.enums.PosId.MAGIC_R  :(2,3), c.enums.PosId.MAGIC_R_R  :(2,2),
                                                 c.enums.PosId.FIELD       :(3,7)
                                                 }
                     }
        for player_id in range(c.enums.PlayerId.UPPER_VALUE):
            for pos_id in range(c.enums.PosId.FIELD+1):
                label = CardLabel(self.master, bg_tkimg, self.udi_gui_frame)
                label_pos = row_col[player_id][pos_id]
                
                # 自分のEXゾーンのときにgridで配置して，boardには自分，相手の場所に同一のlabelを格納する．
                if pos_id in [c.enums.PosId.EX_L_MONSTER, c.enums.PosId.EX_R_MONSTER]:
                    if player_id == c.enums.PlayerId.MYSELF:
                        # 自分のforループのときは，EXゾーンに配置
                        label.grid(row=label_pos[0], column=label_pos[1])
                        self.board[c.enums.PlayerId.MYSELF][pos_id] = label
                        # 相手のEXゾーンは左右反対
                        if pos_id == c.enums.PosId.EX_L_MONSTER:
                            self.board[c.enums.PlayerId.RIVAL][c.enums.PosId.EX_R_MONSTER] = label
                        elif pos_id == c.enums.PosId.EX_R_MONSTER:
                            self.board[c.enums.PlayerId.RIVAL][c.enums.PosId.EX_L_MONSTER] = label
                    else:
                        # 相手のforループのときは，EXゾーンになにもしない
                        pass
                else:
                    label.grid(row=label_pos[0], column=label_pos[1])
                    self.board[player_id][pos_id] = label
        
        # カードリスト関連
        self.card_list_position = {c.enums.PlayerId.MYSELF : dict(), c.enums.PlayerId.RIVAL : dict()}
        row_col = {c.enums.PlayerId.MYSELF:{c.enums.PosId.EXCLUDE:(8,4), c.enums.PosId.GRAVE:(8,5), c.enums.PosId.DECK:(8,6), c.enums.PosId.EXTRA:(8,7)},
                   c.enums.PlayerId.RIVAL :{c.enums.PosId.EXCLUDE:(0,4), c.enums.PosId.GRAVE:(0,5), c.enums.PosId.DECK:(0,6), c.enums.PosId.EXTRA:(0,7)} }
        for player_id in range(c.enums.PlayerId.UPPER_VALUE):
            for pos_id in range(c.enums.PosId.EXTRA, c.enums.PosId.EXCLUDE+1):
                label = PositionLabel(self.master, bg_tkimg, player_id, pos_id, self.udi_gui_frame)
                label_pos = row_col[player_id][pos_id]
                label.grid(row=label_pos[0], column=label_pos[1])
                self.card_list_position[player_id][pos_id] = label

        # RIVAL関連
        self.rival_text_label = tk.Label(self.master, text="相手", font=Const.BOARD_PLAYER_TEXT_FONT,relief="ridge",bg = "lightgray")
        self.rival_text_label.grid(row=0, column=1,pady=4)

        self.rival_lp_label = tk.Label(self.master, text="LP\n8000", font=Const.BOARD_PLAYER_LP_FONT,relief="ridge",bg = "lightgray")
        self.rival_lp_label.grid(row=0, column=2,pady=4)

        self.rival_info_label = tk.Label(self.master, text="召喚権:1", font=Const.BOARD_PLAYER_INFO_FONT,relief="ridge",bg = "lightgray")
        self.rival_info_label.grid(row=0, column=3,pady=4)
        
        # MYSELF関連
        self.myself_text_label = tk.Label(self.master, text="自分", font=Const.BOARD_PLAYER_TEXT_FONT,relief="ridge",bg = "lightgray")
        self.myself_text_label.grid(row=8, column=1,pady=4)

        self.myself_lp_label = tk.Label(self.master, text="LP\n8000", font=Const.BOARD_PLAYER_LP_FONT,relief="ridge",bg = "lightgray")
        self.myself_lp_label.grid(row=8, column=2,pady=4)

        self.myself_info_label = tk.Label(self.master, text="召喚権:1", font=Const.BOARD_PLAYER_INFO_FONT,relief="ridge",bg = "lightgray")
        self.myself_info_label.grid(row=8, column=3,pady=4)

        # 共通関連
        self.phase_label = tk.Label(self.master, text="フェーズ:\n \n ", font=Const.BOARD_PHASE_FONT,relief="ridge",bg = "lightgray")
        self.phase_label.grid(row=9, column=3,columnspan=3,pady=4)


    def reset_board(self):
        for player_id in range(c.enums.PlayerId.UPPER_VALUE):
            for pos_id in range(c.enums.PosId.FIELD + 1):
                self.board[player_id][pos_id].reset()

    def set_image(self, player_id, pos_id, index, card, table_index, overlay_text):
        if index > 0:
            return
        img = self.udi_gui_frame.small_image_manager.get_image_by_card(card)
        # board_managerでのみ，相手のカードは上下反転する
        if player_id == c.enums.PlayerId.RIVAL:
            img = img.transpose(I.FLIP_TOP_BOTTOM)
        img = generate_card_overlay_text(img, overlay_text)

        tkimg = Itk.PhotoImage(img)
        self.board[player_id][pos_id].update(tkimg, card, table_index)

    def update(self, duel_state_data : mdl.DuelStateData):
        # 一旦盤面をリセット
        self.reset_board()

        #####################################################################################################
        # 盤面更新
        duel_card_table = duel_state_data.duel_card_table
        card:mdl.DuelCard
        for table_index, card in enumerate(duel_card_table):
            pos_id = card.pos_id
            player_id = card.player_id
            if c.enums.PlayerId.MYSELF<=player_id<= c.enums.PlayerId.RIVAL:
                # pos_idがモンスターゾーン、魔法罠ゾーン、フィールドゾーンのいずれかのカードのみ
                if pos_id <= c.enums.PosId.FIELD:
                    # カードから、情報を抽出
                    overlay_text, _ = generate_text_by_card_abstract(duel_card_table, card)
                    # indexは、そのposの中で上から何番目のカードかに該当する。エクシーズ素材などはindex>=1になりうるので、今はset_image()内でreturnするようになっている
                    card_index = card.card_index
                    # 画像更新
                    self.set_image(player_id, pos_id, card_index, card, table_index, overlay_text)
        
        #####################################################################################################
        # カードリスト関連更新
        for player_id in range(c.enums.PlayerId.UPPER_VALUE):
            for pos_id in range(c.enums.PosId.EXTRA, c.enums.PosId.EXCLUDE + 1):
                card_list = extract_card_list_by_player_pos(duel_card_table, player_id, pos_id)
                card_num = len(card_list)
                # 枚数表示更新
                self.card_list_position[player_id][pos_id].update(card_num)

        #####################################################################################################
        # 数値関連更新
        general_data = duel_state_data.general_data
        which_turn_now = general_data.which_turn_now
        # RIVAL関連
        if which_turn_now == c.enums.PlayerId.RIVAL:
            # RIVALのターンであれば、RIVALテキストの背景を赤くする
            self.rival_text_label.config(bg="#FF4B00")
        else:
            self.rival_text_label.config(bg="lightgray")

        rival_lp = general_data.lp[c.enums.PlayerId.RIVAL]
        rival_lp_text = f"LP\n{rival_lp}"
        rival_summon_num = general_data.summon_num[c.enums.PlayerId.RIVAL]
        rival_info_text = f"召喚権:{rival_summon_num}"

        # 表示更新
        self.rival_lp_label.config(text=rival_lp_text)
        self.rival_info_label.config(text=rival_info_text)
                
        # MYSELF関連
        if which_turn_now == c.enums.PlayerId.MYSELF:
            # MYSELFのターンであれば、MYSELFテキストの背景を青くする
            self.myself_text_label.config(bg="#005AFF")
        else:
            self.myself_text_label.config(bg="lightgray")

        myself_lp = general_data.lp[c.enums.PlayerId.MYSELF]
        myself_lp_text = f"LP\n{myself_lp}"
        myself_summon_num = general_data.summon_num[c.enums.PlayerId.MYSELF]
        myself_info_text = f"召喚権:{myself_summon_num}"

        # 表示更新
        self.myself_lp_label.config(text=myself_lp_text)
        self.myself_info_label.config(text=myself_info_text)

        #####################################################################################################
        # 共通関連
        phase   = general_data.current_phase
        step    = general_data.current_step
        dmgstep = general_data.current_damage_step

        # 描画が崩れるので、phaseだけ工夫
        if phase == c.enums.Phase.NULL:
            phase_text = "フェイズ無し"
        else:
            phase_text = f"{c.enums.Phase(phase)}"
        step_text = f"{c.enums.StepType(step)}"
        dmgstep_text = f"{c.enums.DmgStepType(dmgstep)}"

        # 表示更新
        self.phase_label.config(text=phase_text+"\n"+step_text+"\n"+dmgstep_text)

