#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""テキスト取得に関するUtil"""

import csv
from dataclasses import fields
import os

from .. import models as mdl 
from .. import constants as c
from ..constants import DuelLogType as dlt
from .card import CardUtil

class TextUtil():
    """UDIのデータを説明するテキストを取得するためのクラス"""

    def __init__(self, card_data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/card_data.csv")):
        self.card_data = {}

        with open(card_data_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                card_id = int(row["card_id"])
                key = card_id
                self.card_data[key] = row

    def get_card_name(self, card_id: int) -> str:
        """
        カード名を取得する
        
        Parameters
        ----------
        card_id : int
            名前を取得したいカードのcard_id

        Returns
        -------
        カード名
            
        """
        if card_id == c.CardId.NO_VALUE:
            text = "カード無し"
        elif card_id == c.CardId.UNKNOWN:
            text = "裏側のカード"
        else:
            try:
                text = f"{self.card_data[card_id]['name']}"
            except KeyError:
                text = f"Key Error: {card_id=}"
        return text

    def get_card_sort_name(self, card_id: int) -> str:
        """
        カード名の日本語の読み仮名を取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            カード名の日本語の読み仮名
        """
        try:
            text = f"{self.card_data[card_id]['sort_name']}"
        except KeyError:
            text = f"KeyError: {card_id=}"
        return text

    def get_card_text(self, card_id: int) -> str:
        """
        カードのテキストを取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            カードのテキスト
        """
        text = ""
        if card_id > c.CardId.START:
            try:
                text = f"{self.card_data[card_id]['text']}"
            except KeyError:
                text = f"Key Error: {card_id=}"
        return text

    def get_effect_text(self, effect_card_id: int, effect_no: int) -> str:
        """
        EffectNoに対応する効果テキストを取得する

        Parameters
        ----------
        effect_card_id : int
            効果を発動したカードのcard_id
        effect_no : int
            効果番号（EffectNo）

        Returns
        -------
        str
            効果番号に対応する効果テキスト
        """
        try:
            text = f"{self.card_data[effect_card_id][str(int(effect_no))]}"
        except KeyError:
            text = f"KeyError: {effect_card_id=}, {effect_no=}"
        return text
    
    def get_card_attr(self, card_id: int) -> str:
        """
        カードの属性を取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            属性
        """
        try:
            text = f"{c.Attribute(int(self.card_data[card_id]['attribute']))}"
        except KeyError:
            text = f"KeyError: {card_id=}"
        return text
    
    def get_level(self, card_id: int):
        """
        カードのレベルを取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            レベル
        """
        try:
            text = f"{self.card_data[card_id]['level']}"
        except KeyError:
            text = f"KeyError: {card_id=}"
        return text
    
    def get_species(self, card_id: int) -> str:
        """
        カードの種族を取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            種族
        """
        try:
            text = f"{c.Species(int(self.card_data[card_id]['species']))}"
        except KeyError:
            text = f"KeyError: {card_id=}"
        return text
    
    def get_atk(self, card_id: int) -> str:
        """
        カードに記載されている攻撃力を取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            カードに記載されている攻撃力
        """
        try:
            atk_val = self.card_data[card_id]["atk"]
        except KeyError:
            atk_val = f"KeyError: {card_id=}"
        if atk_val == "-1":
            return "?"
        else:
            return atk_val
    
    def get_def(self, card_id: int) -> str:
        """
        カードに記載されている守備力を取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            カードに記載されている守備力
        """
        try:
            def_val = self.card_data[card_id]["def"]
        except KeyError:
            def_val = f"KeyError: {card_id=}"
        if def_val == "-1":
            return "?"
        else:
            return def_val
    
    def get_frame(self, card_id: int) -> str:
        """
        カードの種類（【】内の種族以外の項目）を取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            カードの種類（【】内の種族以外の項目）
        """
        try:
            text = f"{c.Frame(int(self.card_data[card_id]['frame']))}"
        except KeyError:
            text = f"KeyError: {card_id=}"
        return text
    
    def get_icon(self, card_id: int) -> str:
        """
        魔法・罠カードの種類を取得する。

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            魔法・罠カードの種類
        """
        try:
            frame_id = c.Frame(int(self.card_data[card_id]['frame']))
            if frame_id is c.Frame.MAGIC or frame_id is c.Frame.TRAP:
                text = f"{c.Icon(int(self.card_data[card_id]['icon']))}"
            else:
                text = ""
        except KeyError:
            text = f"KeyError: {card_id=}"
        return text
    
    def get_scale(self, card_id) -> str:
        """
        ペンデュラムスケールを取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            ペンデュラムスケール
        """
        pend_frame = {c.Frame.PEND,
                      c.Frame.PEND_FX,
                      c.Frame.PEND_TUNER,
                      c.Frame.XYZ_PEND,
                      c.Frame.PEND_FLIP,
                      c.Frame.SYNC_PEND,
                      c.Frame.SP_PEND,
                      c.Frame.FUSION_PEND,
                      c.Frame.PEND_N_TUNER,
                      c.Frame.PEND_SPIRIT,
                      c.Frame.RITUAL_PEND}
        try:
            frame_id = c.Frame(int(self.card_data[card_id]['frame']))
            if frame_id in pend_frame:
                text = self.card_data[card_id]['scale']
            else:
                text = ""
        except KeyError:
            text = f"KeyError: {card_id=}"
        return text

    def get_selection_type_text(self, selection_type: int) -> str:
        """
        selection_typeの説明テキストを取得する

        Parameters
        ----------
        selection_type : int
            selection_type

        Returns
        -------
        str
            selection_typeの説明テキスト
        """
        return f"{c.SelectionType(selection_type)}"

    def get_commands_text(self, commands: list[mdl.CommandEntry]) -> str:
        """
        入力要求の選択肢commandsの説明テキストを取得する

        Parameters
        ----------
        commands : list[mdl.CommandEntry]
            command_requestのcommands

        Returns
        -------
        str
            commandsに含まれているcommandそれぞれの説明テキスト
        """

        text = ""

        for i, command in enumerate(commands):
            text += f"{i:>3}:{self.get_positive_fields_of_command_entry(command)}\n" 
            text += f"{self.get_command_entry_text(command)}\n" 
        
        return text

    def get_command_log_entry_text(self, command_log_entry: mdl.CommandLogEntry) -> str:
        """
        command_logの一単位についての説明テキストを取得する

        Parameters
        ----------
        command_log_entry : mdl.CommandLogEntry
            command_logの一単位

        Returns
        -------
        str
            command_logの一単位の説明テキスト
        """
        text = ""
        command = command_log_entry.command
        selection_type = command_log_entry.selection_type
        selection_type_text = self.get_selection_type_text(selection_type)
        selection_id = command_log_entry.selection_id
        selection_id_text = self.get_selection_id_text(selection_id)
        text += f"{selection_type_text=}, {selection_id_text=}\n"
        text += f"{self.get_command_entry_text(command)}\n"
        
        return text

    def get_command_log_text(self, command_log: list[mdl.CommandLogEntry]) -> str:
        """
        command_logの説明テキストを取得する

        Parameters
        ----------
        command_log : list[mdl.CommandLogEntry]
            command_requestのcommand_logのリスト

        Returns
        -------
        str
            command_logの説明テキスト
        """
        text = ""

        text += "############# command_log #############\n"
        # CommandLog列挙
        for log_entry in command_log:
            text += self.get_command_log_entry_text(log_entry)
        text += "#######################################\n"

        return text

    def get_command_request_text(self, command_request: mdl.CommandRequest) -> str:
        """
        入力要求に関する情報command_request全体の説明テキストを取得する

        Parameters
        ----------
        command_request : mdl.CommandRequest
            CommandRequest

        Returns
        -------
        str
            command_requestの説明テキスト
        """
        text = ""
        
        # selectionType
        text += f"selection_type={command_request.selection_type}\n"

        # selectionId（今までのダイアログテキスト）
        if command_request.selection_id > 0:
            text += f"『{self.get_selection_id_text(command_request.selection_id)}』\n"

        # CommandLog列挙
        if command_request.command_log:
            text += self.get_command_log_text(command_request.command_log)

            
        # Command列挙
        text += self.get_commands_text(command_request.commands)
    
        return text

    def get_positive_fields_of_command_entry(self, command: mdl.CommandEntry) -> str:
        """
        行動の選択肢の一つcommand_entryの、値が0以上の有効な値のみを取り出したテキストを取得する

        Parameters
        ----------
        command_entry : mdl.CommandEntry
            command_requestのcommandsに含まれているcommand一つ

        Returns
        -------
        str
            command_entryの値が0以上の有効な値のみを取り出したテキスト
        """
        text = ""
        attributes = {}
        for key, value in command.__dict__.items():
            if value < 0:
                continue

            attributes[key] = value
        
        text = f"CommandEntry({', '.join(f'{key}={value}' for key, value in attributes.items())})"
        return text

    def get_command_entry_text(self, command: mdl.CommandEntry) -> str:
        """
        行動の選択肢commandの中身を説明したテキストを取得する

        Parameters
        ----------
        command : mdl.CommandEntry
            commandsの中のCommandEntry一つ

        Returns
        -------
        str
            commandの説明テキスト
        """
        text = ""

        card_id = command.card_id

        if command.command_type == c.CommandType.DRAW:
            pass

        elif command.command_type == c.CommandType.CHANGE_PHASE:
            return f"{command.phase} に移行"

        elif command.command_type == c.CommandType.ACTIVATE:
            effect_card_id = command.effect_card_id
            effect_number = command.effect_no
            if effect_number > -1:
                text += f"「{self.get_card_name(command.card_id)}」: {self.get_effect_text(effect_card_id, effect_number)}, "
            else:
                text += f"「{self.get_card_name(command.card_id)}」: 発動(効果番号なし), "

        else:
            if card_id != c.CardId.NO_VALUE:
                text += f"「{self.get_card_name(card_id)}」, "
            else: 
                pass

        if command.table_index > -1:
            text += f"[{command.table_index}], "

        if command.player_id > c.PlayerId.NO_VALUE:
            text += f"{command.player_id}, "

        if command.pos_id > c.PosId.NO_VALUE:
            text += f"{command.pos_id}, "
        else:
            pass

        if command.card_index > c.CardIndex.NO_VALUE:
            text += f"{command.card_index}枚目, "

        if command.stand_face > -1 and command.stand_turn > -1:
            if command.stand_face == 0:
                text += "裏側"
            elif command.stand_face == 1:
                text += "表側"
            if command.stand_turn == 0:
                text += "攻撃表示, "
            if command.stand_turn == 1:
                text += "守備表示, "

        if command.yes_no > c.YesNo.NO_VALUE:
            text += "「はい」" if command.yes_no == 1 else "「いいえ」"

        if command.coin_face > c.Face.NO_VALUE:
            text += "「表」" if command.coin_face == 1 else "「裏」"

        if command.card_attribute > c.Attribute.NO_VALUE:
            text += f"「属性選択:{command.card_attribute}」"

        if command.species > c.Species.NO_VALUE:
            text += f"「種族選択:{command.species}」"

        if command.dialog_text_id > c.TextId.NO_VALUE:
            text += f"「{self.get_dialog_text(command.dialog_text_id)}」"

        if command.number > -1:
            text += f"「数値選択:{command.number}」"

        text += f"{command.command_type}"
        return text

    def get_dialog_text(self, dialog_text_id: int) -> str:
        """
        text_idからそのテキストを取得する

        Parameters
        ----------
        dialog_text_id : int
            text_id

        Returns
        -------
        str
            text_idによって表されるテキスト
        """
        return f"{c.TextId(dialog_text_id)}"

    def get_selection_id_text(self, selection_id: int) -> str:
        """
        選択の種類を説明するテキストを取得する

        Parameters
        ----------
        selection_id : int
            selection_id

        Returns
        -------
        str
            selection_idを説明するテキスト
        """
        if selection_id < 0:
            return ""
        
        return f"{c.SelectionId(selection_id).name}: {c.SelectionId(selection_id)}"

    def get_general_data_text(self, general_data: mdl.GeneralData) -> str:
        """
        デュエル中のカードによらない情報を説明するテキストを取得する

        Parameters
        ----------
        general_data : mdl.GeneralData
            duel_state_dataのgeneral_data

        Returns
        -------
        str
            general_dataを説明するテキスト
        """
        text = ""
        text += f"lp:{general_data.lp}\n"
        text += f"turn_num={general_data.turn_num}（{general_data.turn_num+1}ターン目）\n"
        text += f"{general_data.which_turn_now}のターン\n"
        text += f"フェイズ:{general_data.current_phase}\n"
        text += f"バトルフェイズのステップ:{general_data.current_step}\n"
        text += f"ダメージステップの段階:{general_data.current_damage_step}\n"
        text += f"召喚可能回数{general_data.summon_num}"
        return text

    def get_duel_card_text(self, duel_card: mdl.DuelCard) -> str:
        """
        duel_cardの値が0以上のものに関するテキストを取得する

        Parameters
        ----------
        duel_card : mdl.DuelCard
            duel_state_data.duel_card_taleのduel_card一つ
        """
        text = ""
        attributes = {}
        for key, value in duel_card.__dict__.items():
            if value < 0:
                continue

            # わかりやすさのためにcard_idの時は名前も追加する
            if key == "card_id":
                attributes["card_id"] = f"{value}「{self.get_card_name(value)}」"
            else:
                attributes[key] = value 

        
        text = f"DuelCard({', '.join(f'{key}={value}' for key, value in attributes.items())})"
        return text
    
    def get_battle_position_text(self, face: int, turn: int) -> str:
        """
        モンスターの表示形式を表すテキストを取得する

        Parameters
        ----------
        face : int
            カードの表裏（Face）
        turn : int
            カードの縦横（Turn）

        Returns
        -------
        str
            表示形式
        """
        text = ""
        if face == 1:
            text += "表側"
        elif face == 0:
            text += "裏側"

        if turn == 1:
            text += "守備表示"
        elif turn == 0:
            text += "攻撃表示"

        return text
    
    def get_duel_log_entry_text(self, duel_log_data_entry: mdl.DuelLogDataEntry) -> str:
        """
        デュエルのログの一単位DuelLogDataEntryを説明するテキストを取得する

        Parameters
        ----------
        duel_log_data_entry : mdl.DuelLogDataEntry
            duel_log_dataの中のduel_log_data_entry一つ

        Returns
        -------
        str
            duel_log_data_entryを説明するテキスト
        """
        text = ""

        lt = duel_log_data_entry.type
        ld = duel_log_data_entry.data

        if lt == dlt.DUEL_START:
            text += "デュエル開始"
        
        elif lt == dlt.DUEL_END:
            d = mdl.DuelEndData(ld)
            text += f"{d.finish_type}によって{d.result_type}"

        elif lt == dlt.BATTLE_ATTACK:
            d = mdl.BattleAttackData(ld)
            attack_str = ""
            if d.is_direct_attack > 0:
                attack_str = "ダイレクトアタック"
            else:
                attack_str = f"{d.dst_player_id}の{d.dst_pos_id}の「{self.get_card_name(d.dst_card_id)}」に攻撃"
            text += f"{d.src_player_id}の{d.src_pos_id}の「{self.get_card_name(d.src_card_id)}」が{attack_str}"

        elif lt == dlt.PHASE_CHANGE:
            d = mdl.PhaseChangeData(ld)
            text += f"{d.player_id}の{d.phase}に移行"
        
        elif lt == dlt.TURN_CHANGE:
            d = mdl.TurnChangeData(ld)
            text += f"{d.player_id}のターン"
        
        elif lt == dlt.LIFE_SET:
            d = mdl.LifeSetData(ld)
            text += f"{d.player_id}のLPが{d.life_point}になった"
        
        elif lt == dlt.LIFE_DAMAGE:
            d = mdl.LifeDamageData(ld)
            is_damage = "-" if d.is_damage > 0 else "+"
            text += f"{d.player_id}のLPに{is_damage}{d.damage_val}の{d.damage_type}" 
        
        elif lt == dlt.HAND_OPEN:
            d = mdl.HandOpenData(ld)
            text += f"{d.player_id}が手札の「{self.get_card_name(d.card_id)}」を{d.open_type}"

        elif lt == dlt.DECK_FLIP_TOP:
            d = mdl.DeckFlipTopData(ld)
            open_str = "裏にする" if d.is_open == 0 else "表にする"
            text += f"{d.player_id}の{d.pos_id}の一番上の「{self.get_card_name(d.card_id)}」を{open_str}"

        elif lt == dlt.CARD_LOCKON:
            d = mdl.CardLockonData(ld)
            if d.lockon_type == c.LockonType.ZONE:
                text += f"{d.player_id}の{d.pos_id}を選択"
            else:    
                text += f"{d.player_id}の{d.pos_id}の「{self.get_card_name(d.card_id)}」を{d.lockon_type}"

        elif lt == dlt.CARD_MOVE:
            d = mdl.CardMoveData(ld)
            from_str = f"{d.from_player_id}の{d.from_pos_id}"
            to_str = f"{d.to_player_id}の{d.to_pos_id}"
            text += f"{from_str}の「{self.get_card_name(d.card_id)}」が{d.move_type}によって{to_str}に移動"

        elif lt == dlt.CARD_SWAP:
            d = mdl.CardSwapData(ld)
            text += f"{d.from_player_id}の{d.from_pos_id}のカードと{d.to_player_id}の{d.to_pos_id}のカードを交換"

        elif lt == dlt.CARD_FLIP_TURN:
            d = mdl.CardFlipTurnData(ld)
            if d.pos_id < c.PosId.MAGIC_L_L:
                text += f"{d.player_id}の{d.pos_id}の「{self.get_card_name(d.card_id)}」の表示形式が{self.get_battle_position_text(d.face, d.turn)}に変更"
            else:
                text += f"{d.player_id}の{d.pos_id}の「{self.get_card_name(d.card_id)}」が{d.face}に変更"

        elif lt == dlt.CARD_GENERATE:
            d = mdl.CardGenerateData(ld)
            text += f"{d.player_id}の{d.pos_id}に「{self.get_card_name(d.card_id)}」が{self.get_battle_position_text(d.face, d.turn)}で出現"

        elif lt == dlt.CARD_HAPPEN:
            d = mdl.CardHappenData(ld)
            pos_text = f"{d.pos_id}の" if d.pos_id > c.PosId.NO_VALUE else ""
            text += f"{d.player_id}の{pos_text}「{self.get_card_name(d.card_id)}」"
            if d.effect_no > 0:
                text += f"の「{self.get_effect_text(d.card_id, d.effect_no)}」"
            happen = "適用" if d.is_apply > 0 else "発動"
            text += f"の効果が{happen}"
            
        elif lt == dlt.CARD_DISABLE:
            d = mdl.CardDisableData(ld)
            text += f"{d.player_id}の{d.pos_id}の「{self.get_card_name(d.card_id)}」が無効になった"

        elif lt == dlt.CARD_EQUIP:
            d = mdl.CardEquipData(ld)
            text += f"{d.src_player_id}の{d.src_pos_id}の「{self.get_card_name(d.card_id)}」が{d.dst_player_id}の{d.dst_pos_id}のモンスターに装備された"

        elif lt == dlt.CARD_INC_TURN:
            d = mdl.CardIncTurnData(ld)
            text = f"{d.player_id}の{d.pos_id}の「{self.get_card_name(d.card_id)}」が{d.num}ターン経過"

        elif lt == dlt.COUNTER_SET:
            d = mdl.CounterSetData(ld)
            add_text = "増加" if d.is_add else "減少"
            text = f"{d.player_id}の{d.pos_id}のカウンター({d.counter_type})が{d.add_val}{add_text}"

        elif lt == dlt.MONST_SHUFFLE:
            d = mdl.MonstShuffleData(ld)
            pos_ids_text = [f"{pos_id}" for pos_id in d.pos_ids]
            text = f"{d.player_id}の{pos_ids_text}のカードがシャッフルされた"

        elif lt == dlt.CHAIN_SET:
            d = mdl.ChainSetData(ld)
            text += f"{d.player_id}の{d.pos_id}の「{self.get_card_name(d.card_id)}」がチェーン{d.chain_num}に積まれた"

        elif lt == dlt.CHAIN_RUN:
            d = mdl.ChainRunData(ld)
            text += f"チェーン{d.chain_num}の{d.player_id}の{d.pos_id}の「{self.get_card_name(d.card_id)}」が処理開始"

        elif lt == dlt.RUN_DIALOG:
            d = mdl.RunDialogData(ld)
            if d.text_id > c.TextId.NO_VALUE:
                text += f"「{self.get_dialog_text(d.text_id)}」"
            elif d.selected_text_id > c.TextId.NO_VALUE:
                text += f"「{self.get_dialog_text(d.selected_text_id)}」が選択された"
            elif d.selected_card_id > c.CardId.NO_VALUE:
                if d.selected_effect_no == c.EffectNo.NO_VALUE:
                    text += f"「{self.get_card_name(d.selected_card_id)}」が選択された"
                else:
                    text += f"「{self.get_effect_text(d.selected_card_id, d.selected_effect_no)}」が選択された"
            elif d.selected_numbers:
                text += f"「{d.selected_numbers}」が選択された"
            elif d.selected_species > c.Species.NO_VALUE:
                text += f"{d.selected_species}が選択された"
            elif d.selected_attrs:
                text += f"{[f'{c.Attribute(attr)}' for attr in d.selected_attrs]}が選択された"

        elif lt == dlt.RUN_SUMMON:
            d = mdl.RunSummonData(ld)
            text += f"{d.player_id}の{d.pos_id}に「{self.get_card_name(d.card_id)}」が{self.get_battle_position_text(d.face, d.turn)}で{d.summon_type}された"

        elif lt == dlt.RUN_SP_SUMMON:
            d = mdl.RunSpSummonData(ld)
            text += f"{d.player_id}の{d.pos_id}に「{self.get_card_name(d.card_id)}」が{self.get_battle_position_text(d.face, d.turn)}で特殊召喚された"

        elif lt == dlt.RUN_COIN:
            d = mdl.RunCoinData(ld)
            faces_text = ["表" if f == 1 else "裏" for f in d.faces]
            text += f"コインを{d.num}枚投げて{faces_text}が出た"

        elif lt == dlt.RUN_DICE:
            d = mdl.RunDiceData(ld)
            text += f"{d.player_id}がサイコロを投げて{d.dice}が出た"

        elif lt == dlt.CHAIN_END:
            d = mdl.ChainEndData(ld)
            text += f"{d.player_id}の{d.pos_id}の「{self.get_card_name(d.card_id)}」の処理終了"

        elif lt == dlt.CHAIN_STEP:
            d = mdl.ChainStepData(ld)
            text += f"{d.chain_player_id}の{d.chain_pos_id}の「{self.get_card_name(d.card_id)}」のチェーンブロックが有効に処理開始"

        elif lt == dlt.RUN_FUSION:
            d = mdl.RunFusionData(ld)
            mat_names = [self.get_card_name(mat) for mat in d.material_card_ids]
            smn_names = [self.get_card_name(smn) for smn in d.summon_card_ids]
            if d.fusion_type == c.FusionType.PENDULUM:
                text += f"{d.player_id}が{mat_names}をペンデュラムスケールとして{smn_names}をペンデュラム召喚"
            else:
                text += f"{d.player_id}が{mat_names}を素材として{smn_names}を{d.fusion_type}召喚"

        elif lt == dlt.BATTLE_RUN:
            d = mdl.BattleRunData(ld)
            src_break = "破壊される" if d.is_src_break else "破壊されない"
            text += f"攻撃側:[{src_break}, {d.src_damage}ダメージ]"
            dst_break = "破壊される" if d.is_dst_break else "破壊されない"
            text += f"攻撃対象側:[{dst_break}, {d.dst_damage}ダメージ]"
            text += "の戦闘"
            
        else:
            text += f"{lt.name}: {ld}"

        return text
    
    def get_chain_data_text(self, chain_data: mdl.ChainData) -> str:
        """
        チェーンの情報の一単位ChainDataを説明したテキストを取得する

        Parameters
        ----------
        chain_data : mdl.ChainData
            ChainData一つ

        Returns
        -------
        str
            ChainDataを説明するテキスト
        """
        text = f"====== チェーン{chain_data.chain_num} ======\n"
        text += f"状態={chain_data.chain_state}\n"
        text += f"「{self.get_card_name(chain_data.card_id)}」\n"
        text += f"『{self.get_effect_text(chain_data.card_id, chain_data.effect_no)}』\n"
        text += f"table_index={chain_data.table_index}\n"
        text += f"対象のtable_index={chain_data.target_table_index_list}\n"
        text += f"======================="
        return text


    def get_chain_stack_text(self, chain_stack: list[mdl.ChainData]) -> str:
        """
        チェーンの情報の全体chain_stackを説明したテキストを取得する

        Parameters
        ----------
        chain_stack : list[mdl.ChainData]
            chain_stack

        Returns
        -------
        str
            chain_stackを説明するテキスト
        """
        text_list = []
        for chain_data in chain_stack:
            text_list.append(self.get_chain_data_text(chain_data))

        return "\n".join(text_list) 
    
    def get_general_data_markdown(self, general_data: mdl.GeneralData) -> str:
        """
        GeneralDataをマークダウンの表形式にしたテキストを取得する

        Parameters
        ----------
        general_data : mdl.GeneralData
            GeneralData

        Returns
        -------
        str
            GeneralDataがマークダウンの表形式になったテキスト
        """
        text = ""
        field_names = [field.name for field in fields(mdl.GeneralData)]
        
        header = "| " + " | ".join(field_names) + " |\n"
        separator =  "| " + " | ".join(["---"] * len(field_names)) + " |\n"

        text += header
        text += separator

        values = []
        for field_name in field_names:
            value = str(getattr(general_data, field_name))
            values.append(value)
        text += " | " + " | ".join(values) + " |\n"

        return text

    def get_duel_card_table_markdown(self, duel_card_table: list[mdl.DuelCard]) -> str:
        """
        DuelCardTableをマークダウンの表形式のテキストにしたものを取得する。
        カードが存在しない部分と相手のデッキのカードは省略される。

        Parameters
        ----------
        duel_card_table : list[mdl.DuelCard]
            duel_card_table

        Returns
        -------
        str
            DuelCardTableがマークダウンの表形式になったテキスト
        """
        text = ""
        field_names = [field.name for field in fields(mdl.DuelCard)]
        
        header = "| table_index | " + " | ".join(field_names) + " |\n"
        separator = "| --- | " + " | ".join(["---"] * len(field_names)) + " |\n"

        text += header
        text += separator

        for i, card in enumerate(duel_card_table):
            # プレイヤーが定まっていない（カードがない）カードは出力しない
            if card.card_id == c.CardId.NO_VALUE:
                continue
            values = []
            for field_name in field_names:
                value = str(getattr(card, field_name))
                if field_name == "card_id":
                    card_id = int(value)
                    value = f"{value}({self.get_card_name(card_id)})"
                values.append(value)
            text += f"| {i} | " + " | ".join(values) + " |\n"

        return text
    
    def get_chain_stack_markdown(self, chain_stack: list[mdl.ChainData]) -> str:
        """
        chain_stackをマークダウンの表形式にしたテキストを取得する

        Parameters
        ----------
        chain_stack : list[mdl.ChainData]
            chain_stack

        Returns
        -------
        str
            chain_stackがマークダウンの表形式になったテキスト
        """
        text = ""
        field_names = [field.name for field in fields(mdl.ChainData)]
        
        header = "| index | " + " | ".join(field_names) + " |\n"
        separator = "| --- | " + " | ".join(["---"] * len(field_names)) + " |\n"

        text += header
        text += separator

        for i, chain in enumerate(chain_stack):
            values = []
            for field_name in field_names:
                attr = getattr(chain, field_name)
                if field_name == "card_id":
                    card_id = int(attr)
                    value = f"{attr}({self.get_card_name(card_id)})"
                elif field_name == "effect_no":
                    effect_no = c.EffectNo(attr)
                    effect_text = self.get_effect_text(chain.card_id, effect_no)
                    value = f"{attr}({effect_text})"
                else:
                    value = str(attr)
                values.append(value)
            text += f"| {i} | " + " | ".join(values) + " |\n"

        return text

    def get_commands_markdown(self, commands: list[mdl.CommandEntry]) -> str:
        """
        commandsをマークダウンの表形式にしたテキストを取得する

        Parameters
        ----------
        commands : list[mdl.CommandEntry]
            commands

        Returns
        -------
        str
            commandsがマークダウンの表形式になったテキスト
        """
        text = ""
        field_names = [field.name for field in fields(mdl.CommandEntry)]
        
        header = "| index | " + " | ".join(field_names) + " |\n"
        separator = "| --- | " + " | ".join(["---"] * len(field_names)) + " |\n"

        text += header
        text += separator

        for i, command in enumerate(commands):
            values = []
            for field_name in field_names:
                attr = getattr(command, field_name)
                if field_name == "card_id":
                    card_id = int(attr)
                    value = f"{attr}({self.get_card_name(card_id)})"
                else:
                    value = str(attr)
                values.append(value)
            text += f"| {i} | " + " | ".join(values) + " |\n"

        return text
    
    def get_command_log_markdown(self, command_log: list[mdl.CommandLogEntry]) -> str:
        """
        command_logをマークダウンの表形式にしたテキストを取得する

        Parameters
        ----------
        command_log : list[mdl.CommandLogEntry]
            command_log

        Returns
        -------
        str
            command_logがマークダウンの表形式になったテキスト
        """
        text = ""
        field_names = [field.name for field in fields(mdl.CommandLogEntry)]
        
        header = "| index | " + " | ".join(field_names) + " |\n"
        separator = "| --- | " + " | ".join(["---"] * len(field_names)) + " |\n"

        text += header
        text += separator

        for i, entry in enumerate(command_log):
            values = []
            for field_name in field_names:
                attr = getattr(entry, field_name)
                if field_name == "command":
                    command = attr
                    value = f"{self.get_command_entry_text(command)}"
                else:
                    value = str(attr)
                values.append(value)
            text += f"| {i} | " + " | ".join(values) + " |\n"

        return text