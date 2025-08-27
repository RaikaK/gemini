#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""DuelLogのDuelLogTypeごとのデータの定義"""

from .. import constants as c 
from dataclasses import dataclass 


@dataclass
class DuelStartData:
    """
    デュエル開始。追加データは特になし
    """


@dataclass
class DuelEndData:
    """
    デュエル終了
    """
    result_type: int
    """勝敗タイプ"""
    finish_type: int
    """決着タイプ"""

    def __init__(self, data):
        self.result_type = c.ResultType(data['resultType'])
        self.finish_type = c.FinishType(data['finishType'])


@dataclass
class CardMoveData:
    """
    カード移動
    """
    move_type: int
    """移動タイプ(MoveType参照)"""
    card_id: int
    """移動中のcardId"""
    from_player_id: int
    """移動元のPlayerId"""
    from_pos_id: int
    """移動元のPosId"""
    to_player_id: int
    """移動先のPlayerId"""
    to_pos_id: int
    """移動先のPosId"""

    def __init__(self, data):
        self.move_type = c.MoveType(data['moveType'])
        self.card_id = data['cardId']
        self.from_player_id = c.PlayerId(data['fromPlayerId'])
        self.from_pos_id = c.PosId(data['fromPosId'])
        self.to_player_id = c.PlayerId(data['toPlayerId'])
        self.to_pos_id = c.PosId(data['toPosId'])


@dataclass
class PhaseChangeData:
    """
    フェイズ変更
    """
    player_id: int
    """変更先のフェイズのPlayerId"""
    phase: int
    """変更先のPhase"""

    def __init__(self, data):
        self.player_id = c.PlayerId(data['playerId'])
        self.phase = c.Phase(data['phase'])


@dataclass
class TurnChangeData:
    """
    ターン変更
    """
    player_id: int
    """ターン移動先のPlayerId(PlayerIdのターンになる)"""

    def __init__(self, data):
        self.player_id = c.PlayerId(data['playerId'])


@dataclass
class BattleAttackData:
    """
    攻撃宣言
    """
    src_player_id: int
    """攻撃元のPlayerId"""
    src_pos_id: int
    """攻撃元のPosId"""
    src_card_id: int
    """攻撃元のcardId"""
    dst_player_id: int
    """攻撃先のPlayerId"""
    dst_pos_id: int
    """攻撃先のPosId"""
    dst_card_id: int
    """攻撃先のcardId"""
    is_direct_attack: int
    """ダイレクトアタックかどうか"""

    def __init__(self, data):
        self.src_player_id = c.PlayerId(data['srcPlayerId'])
        self.src_pos_id = c.PosId(data['srcPosId'])
        self.src_card_id = data['srcCardId']
        self.dst_player_id = c.PlayerId(data['dstPlayerId'])
        self.dst_pos_id = c.PosId(data['dstPosId'])
        self.dst_card_id = data['dstCardId']
        self.is_direct_attack = data['isDirectAttack']


@dataclass
class LifeSetData:
    """
    ライフ設定
    """
    player_id: int
    """ライフを設定するPlayerId"""
    life_point: int
    """設定するライフポイントの数値"""

    def __init__(self, data):
        self.player_id = c.PlayerId(data['playerId'])
        self.life_point = data['lifePoint']


@dataclass
class LifeDamageData:
    """
    ライフ増減
    """
    player_id: int
    """ライフが増減するプレイヤーのPlayerId"""
    damage_val: int
    """増減の絶対値"""
    is_damage: int
    """ダメージかどうか（1の場合はダメージ、0の場合は回復）"""
    damage_type: int
    """ダメージの種類(DamageType参照)"""

    def __init__(self, data):
        self.player_id = c.PlayerId(data['playerId'])
        self.damage_val = data['damageVal']
        self.is_damage = data['isDamage']
        self.damage_type = c.DamageType(data['damageType'])


@dataclass
class HandOpenData:
    """
    手札公開
    """
    card_id: int
    """公開されたcardId"""
    player_id: int
    """手札を公開するPlayerId"""
    open_type: int
    """OpenType（ただし裏にするは送られない）"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.open_type = c.OpenType(data['openType'])


@dataclass
class DeckFlipTopData:
    """
    デッキの一番上をめくる
    """
    card_id: int
    """めくられたカードのcardId"""
    player_id: int
    """めくるPlayerId"""
    pos_id: int
    """めくられるデッキのPosId"""
    is_open: int
    """公開かどうか（1なら表、0なら裏になる）"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.is_open = data['isOpen']


@dataclass
class CardLockonData:
    """
    カード選択
    """
    card_id: int
    """選択されるcardId"""
    player_id: int
    """カードのPlayerId"""
    pos_id: int
    """カードのPosId"""
    lockon_type: int
    """選択の種類（LockonType参照）"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.lockon_type = c.LockonType(data['lockonType'])


@dataclass
class CardSwapData:
    """
    フィールド上でカード交換
    """
    from_player_id: int
    """移動元のPlayer"""
    from_pos_id: int
    """移動元のPosId"""
    to_player_id: int
    """移動先のPlayer"""
    to_pos_id: int
    """移動先のPosId"""

    def __init__(self, data):
        self.from_player_id = c.PlayerId(data['fromPlayerId'])
        self.from_pos_id = c.PosId(data['fromPosId'])
        self.to_player_id = c.PlayerId(data['toPlayerId'])
        self.to_pos_id = c.PosId(data['toPosId'])


@dataclass
class CardFlipTurnData:
    """
    カードの表裏攻守変更
    """
    card_id: int
    """変更されるカードのcardId"""
    player_id: int
    """カードのPlayerId"""
    pos_id: int
    """カードのPosId"""
    face: int
    """表裏"""
    turn: int
    """攻守"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.face = c.Face(data['face'])
        self.turn = c.Turn(data['turn'])


@dataclass
class CardGenerateData:
    """
    カード（トークン）の出現
    """
    card_id: int
    """出現するcardId"""
    player_id: int
    """出現する場所のPlayerId"""
    pos_id: int
    """出現する場所のPosId"""
    face: int
    """表裏"""
    turn: int
    """攻守"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.face = c.Face(data['face'])
        self.turn = c.Turn(data['turn'])


@dataclass
class CardHappenData:
    """
    カード効果発動・効果適用
    """
    card_id: int
    """発動・適用するcardId"""
    player_id: int
    """発動した場所のPlayerId"""
    pos_id: int
    """発動した場所のPosId"""
    is_apply: int
    """効果適用か（1なら適用）"""
    effect_no: int
    """効果番号（EffectNo参照）"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.is_apply = data['isApply']
        self.effect_no = c.EffectNo(data['effectNo'])


@dataclass
class CardDisableData:
    """
    カード無効
    """
    card_id: int
    """無効になるカードのcardId"""
    player_id: int
    """カードのPlayerId"""
    pos_id: int
    """カードのPosId"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])


@dataclass
class CardEquipData:
    """
    カード装備
    """
    card_id: int
    """装備カードのcardId"""
    src_player_id: int
    """装備元のカードのPlayerId"""
    src_pos_id: int
    """装備元のカードのPosId"""
    dst_player_id: int
    """装備先のカードのPlayerId"""
    dst_pos_id: int
    """装備先のカードのPosId"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.src_player_id = c.PlayerId(data['srcPlayerId'])
        self.src_pos_id = c.PosId(data['srcPosId'])
        self.dst_player_id = c.PlayerId(data['dstPlayerId'])
        self.dst_pos_id = c.PosId(data['dstPosId'])


@dataclass
class CardIncTurnData:
    """
    ターン経過
    """
    card_id: int
    """カウントが増えるカードのcardId"""
    player_id: int
    """カードのPlayerId"""
    pos_id: int
    """カードのPosId"""
    num: int
    """カウント"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.num = data['num']


@dataclass
class CounterSetData:
    """
    カウンター増減
    """
    player_id: int
    """カードのPlayerId"""
    pos_id: int
    """カードのPosId"""
    add_val: int
    """カウンターの増減値の絶対値"""
    is_add: int
    """増加かどうか（1なら増加、0なら減少）"""
    counter_type: int
    """カウンターの種類"""

    def __init__(self, data):
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.add_val = data['addVal']
        self.is_add = data['isAdd']
        self.counter_type = data['counterType']


@dataclass
class MonstShuffleData:
    """
    モンスター、魔法罠シャッフル
    """
    player_id: int
    """シャッフルされるカードのPlayerId"""
    pos_ids: list[int]
    """シャッフルされるカードのPosIdのリスト"""

    def __init__(self, data):
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_ids = [c.PosId(i) for i in data['posIds']]


@dataclass
class ChainSetData:
    """
    チェーンに積まれた
    """
    card_id: int
    """チェーンに積まれたカードのcardId"""
    player_id: int
    """カードのPlayerId"""
    pos_id: int
    """カードのPosId"""
    chain_num: int
    """チェーンの数（チェーン1は1）"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.chain_num = data['chainNum']


@dataclass
class ChainRunData:
    """
    チェーン解決
    """
    card_id: int
    """チェーン解決中のカードのcardId"""
    player_id: int
    """カードのPlayerId"""
    pos_id: int
    """カードのPosId"""
    chain_num: int
    """チェーンの数（チェーン1は1）"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.chain_num = data['chainNum']


@dataclass
class RunDialogData:
    """
    選択された情報の通知
    """
    text_id: int
    """通知自体のtextId"""
    selected_card_id: int
    """選択されたcardId"""
    selected_text_id: int
    """選択されたtextId"""
    selected_species: int
    """選択された種族"""
    selected_numbers: list[int]
    """選択された数値"""
    selected_attrs: list[int]
    """選択された属性"""
    selected_effect_no: int
    """選択された効果番号"""

    def __init__(self, data):
        self.text_id = c.TextId(data['textId'])
        self.selected_card_id = data['selectedCardId']
        self.selected_text_id = c.TextId(data['selectedTextId'])
        self.selected_species = c.Species(data['selectedSpecies'])
        self.selected_numbers = data['selectedNumbers']
        self.selected_attrs = data['selectedAttrs']
        self.selected_effect_no = c.EffectNo(data['selectedEffectNo'])


@dataclass
class RunSummonData:
    """
    召喚
    """
    card_id: int
    """召喚されたカードのcardId"""
    player_id: int
    """カードのPlayerId"""
    pos_id: int
    """カードのPosId"""
    face: int
    """表裏"""
    turn: int
    """攻守"""
    summon_type: int
    """召喚の種類"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.face = c.Face(data['face'])
        self.turn = c.Turn(data['turn'])
        self.summon_type = c.SummonType(data['summonType'])


@dataclass
class RunSpSummonData:
    """
    特殊召喚
    """
    card_id: int
    """特殊召喚されたカードのcardId"""
    player_id: int
    """カードのPlayerId"""
    pos_id: int
    """カードのPosId"""
    face: int
    """表裏"""
    turn: int
    """攻守"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])
        self.face = c.Face(data['face'])
        self.turn = c.Turn(data['turn'])


@dataclass
class RunCoinData:
    """
    コインが投げられた
    """
    num: int
    """投げる個数"""
    faces: list[int]
    """投げたコインの表裏のリスト（num個の要素を持つ）"""

    def __init__(self, data):
        self.num = data['num']
        self.faces = data['faces']


@dataclass
class RunDiceData:
    """
    サイコロが投げられた
    """
    player_id: int
    """サイコロを投げたプレイヤーのPlayerId"""
    dice: int
    """サイコロの出た目"""

    def __init__(self, data):
        self.player_id = c.PlayerId(data['playerId'])
        self.dice = data['dice']


@dataclass
class ChainEndData:
    """
    チェーン処理終了
    """
    card_id: int
    """チェーン処理が終了したカードのcardId"""
    player_id: int
    """カードのPlayerId"""
    pos_id: int
    """カードのPosId"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.player_id = c.PlayerId(data['playerId'])
        self.pos_id = c.PosId(data['posId'])


@dataclass
class ChainStepData:
    """
    チェーンブロックが有効に処理開始
    """
    card_id: int
    """処理が開始したカードのcardId"""
    chain_player_id: int
    """チェーンブロックのPlayerId"""
    chain_pos_id: int
    """チェーンブロックのPosId"""

    def __init__(self, data):
        self.card_id = data['cardId']
        self.chain_player_id = c.PlayerId(data['chainPlayerId'])
        self.chain_pos_id = c.PosId(data['chainPosId'])


@dataclass
class RunFusionData:
    """
    融合、シンクロなどの特殊召喚情報
    """
    player_id: int
    """特殊召喚するプレイヤーのPlayerId"""
    summon_card_ids: list[int]
    """特殊召喚するカードのcardIdのリスト（ペンデュラムの場合は複数）"""
    fusion_type: int
    """特殊召喚の種類"""
    material_card_ids: list[int]
    """素材となるカードのcardIdのリスト（ペンデュラムの場合はスケール）"""

    def __init__(self, data):
        self.player_id = c.PlayerId(data['playerId'])
        self.summon_card_ids = data['summonCardIds']
        self.fusion_type = c.FusionType(data['fusionType'])
        self.material_card_ids = data['materialCardIds']


@dataclass
class BattleRunData:
    """
    戦闘の情報
    """
    src_damage: int
    """攻撃側が受けるダメージの絶対値"""
    dst_damage: int
    """攻撃対象側が受けるダメージの絶対値"""
    is_src_break: int
    """攻撃側が破壊されるかどうか"""
    is_dst_break: int
    """攻撃対象側が破壊されるかどうか"""

    def __init__(self, data):
        self.src_damage = data['srcDamage']
        self.dst_damage = data['dstDamage']
        self.is_src_break = data['isSrcBreak']
        self.is_dst_break = data['isDstBreak']


