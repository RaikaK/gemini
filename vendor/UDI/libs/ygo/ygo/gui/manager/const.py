#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

class Const:
    # main frame
    GEO_MAIN = '1920x1080'
    # DIRサイズ
    RIGHT_DIR_WIDTH = 170
    MID_DIR_WIDTH = 670
    ADDITIONAL_DIR_WIDTH = 400

    CHAIN_DIR_HEIGHT = 500
    HAND_DIR_HEIGHT = 175
    DIALOG_DIR_HEIGHT = 110
    CARD_TEXT_DIR_HEIGHT = 320
    CONTEXT_DIR_HEIGHT = 400

    # カードサイズ
    S_CARD_H = 90
    S_CARD_W = 90
    M_CARD_H = 125
    M_CARD_W = 125
    L_CARD_H = 200
    L_CARD_W = 200

    # PAD
    C_LIST_PADX = 12
    C_LIST_PADY = 2
    LOG_PADY = 2
    COMMAND_PADY = 2
    COMMAND_BUTTON_PADX = 2
    COMMAND_BUTTON_PADY = 2

    # フォント
    BOARD_POSITION_FONT = ('MSゴシック', '10', "bold")
    BOARD_PLAYER_TEXT_FONT = ('MSゴシック', '20', "bold")
    BOARD_PLAYER_LP_FONT =  ('MSゴシック', '15', "bold")
    BOARD_PLAYER_INFO_FONT =  ('MSゴシック', '10', "bold")
    BOARD_PHASE_FONT = ('MSゴシック', '10', "bold")

    CHAIN_NUM_FONT = ('MSゴシック', '10', "bold")
    CHAIN_TEXT_FONT = ('MSゴシック', '9')
    CHAIN_WRAP_LENGTH = M_CARD_H - 10

    C_LIST_FONT = ('MSゴシック', '10')
    C_LIST_WRAP_LENGTH = M_CARD_H
    C_LIST_INFO_FONT = ('MSゴシック', '13', "bold")
    C_LIST_INFO_WRAP_LENGTH = M_CARD_H

    CONTEXT_NUM_FONT = ('MSゴシック', '10', "bold")
    CONTEXT_TEXT_FONT = ('MSゴシック', '8')
    CONTEXT_SUBTEXT_FONT = ('Helvetica', '8')
    CONTEXT_IMGLABEL_FONT = ('MSゴシック', '10', "bold")
    CONTEXT_WRAP_LENGTH = ADDITIONAL_DIR_WIDTH - S_CARD_W - 10

    LOG_NUM_FONT =  ('MSゴシック', '10', "bold")
    LOG_TEXT_FONT = ('Helvetica', '8')
    LOG_WRAP_LENGTH = ADDITIONAL_DIR_WIDTH - 40

    DIALOG_FONT = ('MSゴシック', '12', "bold")
    DIALOG_WRAP_LENGTH = 1920-RIGHT_DIR_WIDTH-MID_DIR_WIDTH-ADDITIONAL_DIR_WIDTH - 30

    CARDTEXT_FONT = ('Helvetica', '12')
    CARDTEXT_NAME_FONT = ('MSゴシック', '14', "bold")
    CARDTEXT_WRAP_LENGTH = 1920-RIGHT_DIR_WIDTH-MID_DIR_WIDTH-ADDITIONAL_DIR_WIDTH - 20 - L_CARD_W
    
    COMMAND_NUM_FONT = ('MSゴシック', '10', "bold")
    COMMAND_TEXT_FONT = ('MSゴシック', '10', "bold")
    COMMAND_SUBTEXT_FONT = ('Helvetica', '8')
    COMMAND_IMGLABEL_FONT = ('MSゴシック', '10', "bold")
    COMMAND_BUTTON_FONT = ('MSゴシック', '15', "bold")
    COMMAND_WRAP_LENGTH = MID_DIR_WIDTH - M_CARD_W - 3 * int(COMMAND_BUTTON_FONT[1]) - 100

    IMAGE_FONT_PATH = 'C:/Windows/Fonts/msgothic.ttc'
    IMAGE_FONT_SIZE = 13

    # ハイライトカラー
    HIGHLIGHT_BLACK = "#000000"
    HIGHLIGHT_GRAY = "#808080"
    DEFAULT_COLOR = "SystemButtonFace"
    HIGHLIGHT_RED = "#ff0000"

    # その他
    CHAIN_BD = 1
    C_LIST_BD = 1
    LOG_BD = 1


