#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import textwrap
from PIL import Image, ImageDraw, ImageFont

from ygo.udi_io import UdiIO
from ygo import models as mdl
from ygo import constants as c
from ygo.util.text import TextUtil
from ygo.util.card import CardUtil

from .const import Const



def extract_card_list_by_player_pos(duel_card_table, player_id, pos_id):
    if player_id == c.enums.PlayerId.NO_VALUE or pos_id == c.enums.PosId.NO_VALUE:
        return []
    
    card_list = []
    card : mdl.DuelCard
    for table_index, card in enumerate(duel_card_table):
        if card.player_id == player_id and card.pos_id == pos_id:
            card_list.append((table_index, card))
            
    # card_indexでソート
    card_list.sort(key=lambda x: x[1].card_index)
    
    return card_list


def generate_text_by_card_abstract(duel_card_table, card:mdl.DuelCard):
    # カードに重ねて表示するテキスト
    overlay_text = ""
    # card_id, player_id, pos_id, card_index, face, turnは画像を見れば分かるはず

    # 効果無効
    disabled = card.is_disabled
    if disabled == 1:
        overlay_text+="Disabled, "
    # 攻撃関連
    attacking = card.is_attacking
    if attacking == 1:
        overlay_text+="Atking, "
    attacked = card.is_attacked
    if attacked == 1:
        overlay_text+="Atked, "
    #TODO: チェーンスタックからチェーン関連情報を抜き出して表示するか考える(以前：setnum>0であれば、とりあえずチェーンに関連している。abstractでは、チェーンに関連しているかだけ表示しておく)
    
    #TODO: コスト払い中または対象選択中か,効果処理中か HAPPEN_PRE_SELECTING, HAPPEN_EFFECT_PROCESSINGって何だったか確認

    # 魔法罠の経過ターン
    turn_passed = card.turn_passed
    if turn_passed == 1:
        overlay_text+="Passed 1 turn, "

    # カウンター
    counter = card.magic_counter_num
    if counter > 0:
        overlay_text+="MagC:"+str(counter)+", "
    # 装備情報
    eq_target = card.equip_target
    if eq_target != -1:
        eq_target_card:mdl.DuelCard = duel_card_table[eq_target]
        eq_player = eq_target_card.player_id
        eq_pos = eq_target_card.pos_id
        overlay_text+="Eq:P"+str(eq_player)+" pos"+str(eq_pos)+", "

    if overlay_text != "":
        overlay_text+="\n\n"

    # 攻撃力防御力
    card_atk = card.atk_val
    card_def = card.def_val
    if card_atk != -1:
        overlay_text+="ATK:"+str(card_atk)+"\n"
    if card_def != -1:
        overlay_text+="DEF:"+str(card_def)


    # カードの横に表示するテキスト(盤面では表示しない)
    # 効果使用済み関連
    used_ef_list = [
        card.used_effect1,
        card.used_effect2,
        card.used_effect3,
    ]
    side_text = ""
    for i, used in enumerate(used_ef_list):
        if used == 1:
            side_text += "効果"+str(i+1)+"使用済み\n"

    return overlay_text, side_text


def generate_text_by_card_detailed(duel_card_table, card:mdl.DuelCard, card_util:CardUtil):
    # 詳細表示のためのテキスト
    text = ""

    # 効果無効
    disabled = card.is_disabled
    if disabled == 1:
        text+="Effect Disabled, \n"
    # 攻撃関連
    attacking = card.is_attacking
    if attacking == 1:
        text+="Attacking, \n"
    attacked = card.is_attacked
    if attacked == 1:
        text+="Attacked, \n"
    
    #TODO: チェーンスタックからチェーン関連情報を抜き出して表示するか考える(以前：setnum>0であれば、とりあえずチェーンに関連している。abstractでは、チェーンに関連しているかだけ表示しておく)
    
    #TODO: コスト払い中または対象選択中か,効果処理中か HAPPEN_PRE

    # 魔法罠の経過ターン
    turn_passed = card.turn_passed
    if turn_passed == 1:
        text+="Passed 1 turn, \n"
    # カウンター
    counter = card.magic_counter_num
    if counter > 0:
        text+="Magic Counter:"+str(counter)+", \n"
    # 装備情報
    eq_target = card.equip_target
    if eq_target != -1:
        eq_target_card:mdl.DuelCard = duel_card_table[eq_target]
        eq_player = eq_target_card.player_id
        eq_pos = eq_target_card.pos_id
        text+="Equip Target:P"+str(eq_player)+" " + f"{c.enums.PosId(eq_pos)}"
        try:
            text += card_util.get_name(eq_target_card.card_id)
        except KeyError:
            text += "不明"
        text += "\n"

    # 攻撃力防御力
    card_atk = card.atk_val
    card_def = card.def_val
    text += "ATK:"+str(card_atk)
    text += "  "
    text += "DEF:"+str(card_def)+"\n"

    # 効果使用済み関連
    used_ef_list = [
        card.used_effect1,
        card.used_effect2,
        card.used_effect3,
    ]
    for i, used in enumerate(used_ef_list):
        if used == 1:
            text += "効果"+str(i+1)+"使用済み\n"

    return text

# カードのテキスト以外の情報をテキスト化
def generate_card_info_text(card_id, card_util:CardUtil):
    text = ""

    try:
        text += "【"
        text += str(card_util.get_frame(card_id))
        text += "】 "

        text += "【"
        text += str(card_util.get_icon(card_id))
        text += "】 "
        text += "\n"

        text += "属性:"
        text += str(card_util.get_attribute(card_id))
        text += " "

        text += "☆:"
        text += str(card_util.get_level(card_id))
        text += " "

        text += "種族:"
        text += str(card_util.get_species(card_id))
        text += " "

        text += "\n"
        text += "ATK:"
        text += str(card_util.get_atk(card_id))
        text += " "

        text += "DEF:"
        text += str(card_util.get_def(card_id))
        text += " "

        p_scale = card_util.get_scale(card_id)
        if p_scale >= 0:
            text += "\n"
            text += "スケール:"
            text += str(p_scale)
    except KeyError:
        pass

    return text


# GUI用テキスト生成
def make_command_text(command:mdl.CommandEntry):
    command_type   = command.command_type
    card_id        = command.card_id
    card_player_id = command.player_id
    card_position  = command.pos_id
    stand_face     = command.stand_face
    stand_turn     = command.stand_turn
    coin_face      = command.coin_face

    command_text = f"{c.CommandType(command_type)}"
    # 決定
    if command_type == c.enums.CommandType.DECIDE:
        if card_player_id > c.PlayerId.NO_VALUE:
            command_text += f"{c.PlayerId(card_player_id)} "
        if card_position > c.PosId.NO_VALUE:
            command_text += f"{c.enums.PosId(card_position)} "
        if stand_face > -1 and stand_turn > -1:
            if stand_face == 0:
                command_text += "裏側"
            elif stand_face == 1:
                command_text += "表側"
            if stand_turn == 0:
                command_text += "攻撃表示 "
            elif stand_turn == 1:
                command_text += "守備表示 "
        if command.yes_no > -1:
            command_text += "「はい」" if command.yes_no == 1 else "「いいえ」"
        if command.coin_face > -1:
            command_text += "コイン表" if command.coin_face == 1 else "コイン裏"
        if command.dialog_text_id > -1:
            text_util = TextUtil()
            command_text += f"「{text_util.get_dialog_text(command.dialog_text_id)}」"

    # 発動
    if command_type == c.enums.CommandType.ACTIVATE:
        effect_no = command.effect_no
        command_text += f'{c.enums.EffectNo(effect_no)}'
        
    # フェイズ
    elif command_type == c.enums.CommandType.CHANGE_PHASE:
        phase = command.phase
        command_text += f'{c.enums.Phase(phase)}'

    return command_text

# GUI用アイコン生成
def make_command_icon(command:mdl.CommandEntry):
    icon_type = UdiIO.RatingTextType.ETC
    icon_id = 0

    # command_typeで判別可能
    command_type = command.command_type
    # PHASE
    if command_type == c.enums.CommandType.CHANGE_PHASE:
        icon_type = UdiIO.RatingTextType.PHASE
        phase = command.phase
        icon_id = phase
        return (icon_type, icon_id)
    
    # # REJECT
    # if command_type == c.enums.CommandType.CANCEL:
    #     icon_type = UdiIO.RatingTextType.CANCEL
    #     icon_id = 0
    #     return (icon_type, icon_id)
    
    # DRAW
    if command_type == c.enums.CommandType.DRAW:
        icon_type = UdiIO.RatingTextType.DRAW
        icon_id = 0
        return (icon_type, icon_id)

    # command_typeで判別不可
    # COIN
    coin_face = command.coin_face
    if coin_face != -1:
        icon_type = UdiIO.RatingTextType.COIN
        icon_id = coin_face
        return (icon_type, icon_id)
    
    # CARD_ID
    card_id = command.card_id
    if card_id != -1:
        icon_type = UdiIO.RatingTextType.CARD_ID
        icon_id = card_id
        return (icon_type, icon_id)
    
    # POSITION(先にCARD_IDで引っかからないなら場所選択っぽい)
    card_position = command.pos_id
    if card_position != -1:
        if command_type == c.enums.CommandType.DECIDE:
            icon_type = UdiIO.RatingTextType.POSITION
            icon_id = card_position
            return (icon_type, icon_id)

    # ここまできたらETCで返す
    return (icon_type, icon_id)


def generate_card_overlay_text(img:Image.Image, text:str):
    img = img.convert('RGBA')
    w,h = img.size
    font = ImageFont.truetype(Const.IMAGE_FONT_PATH, Const.IMAGE_FONT_SIZE)

    text_img = Image.new('RGBA',(w,h))
    draw = ImageDraw.Draw(text_img)

    lines = []
    for line in text.split('\n'):
        lines.extend(textwrap.wrap(line, width=w // font.getbbox(' ')[2]))
    x = 0+1
    y = h-1
    for line in reversed(lines):
        draw.rectangle(draw.textbbox((x,y) , line, font = font, anchor="ld"), fill=(255,255,255,128))
        draw.text( (x,y) , line, 'red', font = font, anchor="ld")
        y -= font.getbbox(line)[3]

    img = Image.alpha_composite(img, text_img)

    return img
