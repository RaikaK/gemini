#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""入力要求に関する情報の定義"""

from .. import constants as c 
from dataclasses import dataclass 


@dataclass
class CommandEntry:
    """
    コマンドの選択肢の情報
    """
    player_id: int
    """該当カードの場所のPlayerId"""
    pos_id: int
    """該当カードの場所のPosId"""
    card_index: int
    """該当カードの場所のcardIndex"""
    command_type: int
    """コマンドのCommandType"""
    card_id: int
    """該当カードのcardId"""
    effect_card_id: int
    """commandTypeがActivateの場合に、発動する効果が書いてあるカードのcardId"""
    effect_no: int
    """commandTypeがActivateの場合に、発動する効果の効果番号"""
    phase: int
    """移行先のフェイズ"""
    dialog_text_id: int
    """ダイアログからテキストを選択する場合のTextId"""
    stand_face: int
    """表示形式選択の表裏の選択肢"""
    stand_turn: int
    """表示形式選択の攻守の選択肢"""
    coin_face: int
    """コイン選択の選択肢"""
    card_attribute: int
    """属性選択の選択肢"""
    species: int
    """種族選択の選択肢"""
    number: int
    """サイコロなど数値の選択肢"""
    yes_no: int
    """YesNoの選択肢"""
    table_index: int
    """DuelStateDataのduelCardTableにおける該当カードのインデックス"""

    def __init__(self, data):
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.card_index = data['cardIndex']
        self.command_type = c.CommandType(data['commandType'])
        self.card_id = data['cardId']
        self.effect_card_id = data['effectCardId']
        self.effect_no = c.EffectNo(data['effectNo'])
        self.phase = c.Phase(data['phase'])
        self.dialog_text_id = c.TextId(data['dialogTextId'])
        self.stand_face = c.Face(data['standFace'])
        self.stand_turn = c.Turn(data['standTurn'])
        self.coin_face = c.Face(data['coinFace'])
        self.card_attribute = c.Attribute(data['cardAttribute'])
        self.species = c.Species(data['species'])
        self.number = data['number']
        self.yes_no = c.YesNo(data['yesNo'])
        self.table_index = data['tableIndex']


@dataclass
class CommandLogEntry:
    """
    コマンドログの一つの単位
    """
    command: CommandEntry
    """その時選択したCommandEntry"""
    selection_type: int
    """その時のselectionType"""
    selection_id: int
    """その時のselectionId"""

    def __init__(self, data):
        self.command = CommandEntry(data['command'])
        self.selection_type = c.SelectionType(data['selectionType'])
        self.selection_id = c.SelectionId(data['selectionId'])


@dataclass
class CommandRequest:
    """
    入力要求に関する情報
    """
    commands: list[CommandEntry]
    """コマンドの選択肢"""
    selection_type: int
    """入力要求の種類"""
    selection_id: int
    """入力要求に関するテキストのId"""
    command_log: list[CommandLogEntry]
    """入力要求に関する今まで実行したコマンド"""

    def __init__(self, data):
        self.commands = [CommandEntry(command) for command in data['commands']]
        self.selection_type = c.SelectionType(data['selectionType'])
        self.selection_id = c.SelectionId(data['selectionId'])
        self.command_log = [CommandLogEntry(command_log) for command_log in data['commandLog']]


