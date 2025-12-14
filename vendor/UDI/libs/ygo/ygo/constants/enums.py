#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""UDIで使うenumの定義"""

from enum import IntEnum

class CardId(IntEnum):
    """
    カードidにまつわる定数
    """
    NO_VALUE = -1
    """値無し"""
    UNKNOWN = 0
    """裏側のカード"""
    START = 1000
    """card_idの値として有効な値の最低値"""

    def __str__(self):
        descriptions = {
            CardId.NO_VALUE: "値無し",
            CardId.UNKNOWN: "裏側のカード",
            CardId.START: "card_idの値として有効な値の最低値",
        }
        return descriptions.get(self, "不明")

class Face(IntEnum):
    """
    カードの表裏
    """
    NO_VALUE = -1
    """値無し"""
    BACK = 0
    """裏"""
    FRONT = 1
    """表"""
    UPPER_VALUE = 2
    """上限値"""

    def __str__(self):
        descriptions = {
            Face.NO_VALUE: "値無し",
            Face.BACK: "裏",
            Face.FRONT: "表",
            Face.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class Turn(IntEnum):
    """
    カードの縦横（攻守）
    """
    NO_VALUE = -1
    """値無し"""
    VERTICAL = 0
    """縦（攻撃）"""
    HORIZONTAL = 1
    """横（守備）"""
    UPPER_VALUE = 2
    """上限値"""

    def __str__(self):
        descriptions = {
            Turn.NO_VALUE: "値無し",
            Turn.VERTICAL: "縦（攻撃）",
            Turn.HORIZONTAL: "横（守備）",
            Turn.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class Attribute(IntEnum):
    """
    属性（魔法・罠も含む）
    """
    NO_VALUE = -1
    """値無し"""
    NULL = 0
    """属性なし"""
    LIGHT = 1
    """光"""
    DARK = 2
    """闇"""
    WATER = 3
    """水"""
    FIRE = 4
    """炎"""
    EARTH = 5
    """地"""
    WIND = 6
    """風"""
    GOD = 7
    """神"""
    MAGIC = 8
    """魔法"""
    TRAP = 9
    """罠"""
    UPPER_VALUE = 10
    """上限値"""

    def __str__(self):
        descriptions = {
            Attribute.NO_VALUE: "値無し",
            Attribute.NULL: "属性なし",
            Attribute.LIGHT: "光",
            Attribute.DARK: "闇",
            Attribute.WATER: "水",
            Attribute.FIRE: "炎",
            Attribute.EARTH: "地",
            Attribute.WIND: "風",
            Attribute.GOD: "神",
            Attribute.MAGIC: "魔法",
            Attribute.TRAP: "罠",
            Attribute.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class Species(IntEnum):
    """
    種族
    """
    NO_VALUE = -1
    """値無し"""
    NULL = 0
    """種族なし"""
    DRAGON = 1
    """ドラゴン族"""
    ZOMBIE = 2
    """アンデット族"""
    FIEND = 3
    """悪魔族"""
    PYRO = 4
    """炎族"""
    SEA_SERPENT = 5
    """海竜族"""
    ROCK = 6
    """岩石族"""
    MACHINE = 7
    """機械族"""
    FISH = 8
    """魚族"""
    DINOSAUR = 9
    """恐竜族"""
    INSECT = 10
    """昆虫族"""
    BEAST = 11
    """獣族"""
    BEAST_WARRIOR = 12
    """獣戦士族"""
    PLANT = 13
    """植物族"""
    AQUA = 14
    """水族"""
    WARRIOR = 15
    """戦士族"""
    WINGED_BEAST = 16
    """鳥獣族"""
    FAIRY = 17
    """天使族"""
    SPELLCASTER = 18
    """魔法使い族"""
    THUNDER = 19
    """雷族"""
    REPTILE = 20
    """爬虫類族"""
    PSYCHIC = 21
    """サイキック族"""
    WYRM = 22
    """幻竜族"""
    CYBERSE = 23
    """サイバース族"""
    DIVINE_BEAST = 24
    """幻神獣族"""
    ILLUSION = 25
    """幻想魔族"""
    UPPER_VALUE = 26
    """上限値"""

    def __str__(self):
        descriptions = {
            Species.NO_VALUE: "値無し",
            Species.NULL: "種族なし",
            Species.DRAGON: "ドラゴン族",
            Species.ZOMBIE: "アンデット族",
            Species.FIEND: "悪魔族",
            Species.PYRO: "炎族",
            Species.SEA_SERPENT: "海竜族",
            Species.ROCK: "岩石族",
            Species.MACHINE: "機械族",
            Species.FISH: "魚族",
            Species.DINOSAUR: "恐竜族",
            Species.INSECT: "昆虫族",
            Species.BEAST: "獣族",
            Species.BEAST_WARRIOR: "獣戦士族",
            Species.PLANT: "植物族",
            Species.AQUA: "水族",
            Species.WARRIOR: "戦士族",
            Species.WINGED_BEAST: "鳥獣族",
            Species.FAIRY: "天使族",
            Species.SPELLCASTER: "魔法使い族",
            Species.THUNDER: "雷族",
            Species.REPTILE: "爬虫類族",
            Species.PSYCHIC: "サイキック族",
            Species.WYRM: "幻竜族",
            Species.CYBERSE: "サイバース族",
            Species.DIVINE_BEAST: "幻神獣族",
            Species.ILLUSION: "幻想魔族",
            Species.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class Icon(IntEnum):
    """
    アイコン
    """
    NO_VALUE = -1
    """値無し"""
    NULL = 0
    """通常"""
    COUNTER = 1
    """カウンター"""
    FIELD = 2
    """フィールド"""
    EQUIP = 3
    """装備"""
    CONTINUOUS = 4
    """永続"""
    QUICK_PLAY = 5
    """速攻"""
    RITUAL = 6
    """儀式"""
    UPPER_VALUE = 7
    """上限値"""

    def __str__(self):
        descriptions = {
            Icon.NO_VALUE: "値無し",
            Icon.NULL: "通常",
            Icon.COUNTER: "カウンター",
            Icon.FIELD: "フィールド",
            Icon.EQUIP: "装備",
            Icon.CONTINUOUS: "永続",
            Icon.QUICK_PLAY: "速攻",
            Icon.RITUAL: "儀式",
            Icon.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class Phase(IntEnum):
    """
    フェイズ
    """
    NO_VALUE = -1
    """値無し"""
    DRAW = 0
    """ドローフェイズ"""
    STANDBY = 1
    """スタンバイフェイズ"""
    MAIN1 = 2
    """メインフェイズ1"""
    BATTLE = 3
    """バトルフェイズ"""
    MAIN2 = 4
    """メインフェイズ2"""
    END = 5
    """エンドフェイズ"""
    NULL = 7
    """フェイズ無し（TurnChangeのタイミングで来る）"""
    UPPER_VALUE = 8
    """上限値"""

    def __str__(self):
        descriptions = {
            Phase.NO_VALUE: "値無し",
            Phase.DRAW: "ドローフェイズ",
            Phase.STANDBY: "スタンバイフェイズ",
            Phase.MAIN1: "メインフェイズ1",
            Phase.BATTLE: "バトルフェイズ",
            Phase.MAIN2: "メインフェイズ2",
            Phase.END: "エンドフェイズ",
            Phase.NULL: "フェイズ無し（TurnChangeのタイミングで来る）",
            Phase.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class StepType(IntEnum):
    """
    バトルフェイズのステップ
    """
    NO_VALUE = -1
    """値無し"""
    NULL = 0
    """バトルフェイズ以外"""
    START = 1
    """スタートステップ"""
    BATTLE = 2
    """バトルステップ"""
    DAMAGE = 3
    """ダメージステップ"""
    END = 4
    """エンドステップ"""
    UPPER_VALUE = 5
    """上限値"""

    def __str__(self):
        descriptions = {
            StepType.NO_VALUE: "値無し",
            StepType.NULL: "バトルフェイズ以外",
            StepType.START: "スタートステップ",
            StepType.BATTLE: "バトルステップ",
            StepType.DAMAGE: "ダメージステップ",
            StepType.END: "エンドステップ",
            StepType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class DmgStepType(IntEnum):
    """
    ダメージステップの段階
    """
    NO_VALUE = -1
    """値無し"""
    NULL = 0
    """ダメージステップ以外"""
    START = 1
    """ダメージステップ開始時"""
    BEFORE_CALC = 2
    """ダメージ計算前"""
    DAMAGE_CALC = 3
    """ダメージ計算時"""
    AFTER_CALC = 4
    """ダメージ計算後"""
    END = 5
    """ダメージステップ終了時"""
    UPPER_VALUE = 6
    """上限値"""

    def __str__(self):
        descriptions = {
            DmgStepType.NO_VALUE: "値無し",
            DmgStepType.NULL: "ダメージステップ以外",
            DmgStepType.START: "ダメージステップ開始時",
            DmgStepType.BEFORE_CALC: "ダメージ計算前",
            DmgStepType.DAMAGE_CALC: "ダメージ計算時",
            DmgStepType.AFTER_CALC: "ダメージ計算後",
            DmgStepType.END: "ダメージステップ終了時",
            DmgStepType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class ResultType(IntEnum):
    """
    デュエル結果
    """
    NO_VALUE = -1
    """値無し"""
    NONE = 0
    """未決着"""
    WIN = 1
    """勝利"""
    LOSE = 2
    """敗北"""
    DRAW = 3
    """引き分け"""
    OTHER = 4
    """その他エラー等"""
    UPPER_VALUE = 5
    """上限値"""

    def __str__(self):
        descriptions = {
            ResultType.NO_VALUE: "値無し",
            ResultType.NONE: "未決着",
            ResultType.WIN: "勝利",
            ResultType.LOSE: "敗北",
            ResultType.DRAW: "引き分け",
            ResultType.OTHER: "その他エラー等",
            ResultType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class FinishType(IntEnum):
    """
    デュエル勝敗タイプ
    """
    NO_VALUE = -1
    """値無し"""
    NONE = 0
    """未決着"""
    NORMAL = 1
    """ライフを0にした"""
    NO_DECK = 2
    """デッキがなくなった"""
    SURRENDER = 3
    """降参"""
    SPECIAL_WIN = 4
    """特殊勝利"""
    TIME_OUT = 5
    """時間切れ"""
    OTHER = 6
    """その他エラー等"""
    UPPER_VALUE = 7
    """上限値"""

    def __str__(self):
        descriptions = {
            FinishType.NO_VALUE: "値無し",
            FinishType.NONE: "未決着",
            FinishType.NORMAL: "ライフを0にした",
            FinishType.NO_DECK: "デッキがなくなった",
            FinishType.SURRENDER: "降参",
            FinishType.SPECIAL_WIN: "特殊勝利",
            FinishType.TIME_OUT: "時間切れ",
            FinishType.OTHER: "その他エラー等",
            FinishType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class PlayerId(IntEnum):
    """
    プレイヤー番号
    """
    NO_VALUE = -1
    """値無し"""
    MYSELF = 0
    """自分"""
    RIVAL = 1
    """相手"""
    UPPER_VALUE = 2
    """上限値"""

    def __str__(self):
        descriptions = {
            PlayerId.NO_VALUE: "値無し",
            PlayerId.MYSELF: "自分",
            PlayerId.RIVAL: "相手",
            PlayerId.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class PosId(IntEnum):
    """
    カードなどの位置を表す番号
    """
    NO_VALUE = -1
    """値無し"""
    MONSTER_L_L = 0
    """メインモンスターゾーン左端"""
    MONSTER_L = 1
    """メインモンスターゾーン左から2番目"""
    MONSTER_C = 2
    """メインモンスターゾーン中央"""
    MONSTER_R = 3
    """メインモンスターゾーン右から2番目"""
    MONSTER_R_R = 4
    """メインモンスターゾーン右端"""
    EX_L_MONSTER = 5
    """EXモンスターゾーン左"""
    EX_R_MONSTER = 6
    """EXモンスターゾーン右"""
    MAGIC_L_L = 7
    """魔法＆罠ゾーン左端"""
    MAGIC_L = 8
    """魔法＆罠ゾーン左から2番目"""
    MAGIC_C = 9
    """魔法＆罠ゾーン中央"""
    MAGIC_R = 10
    """魔法＆罠ゾーン右から2番目"""
    MAGIC_R_R = 11
    """魔法＆罠ゾーン右端"""
    FIELD = 12
    """フィールドゾーン"""
    HAND = 13
    """手札"""
    EXTRA = 14
    """EXデッキ"""
    DECK = 15
    """デッキ"""
    GRAVE = 16
    """墓地"""
    EXCLUDE = 17
    """除外"""
    DISPLAY = 18
    """表示のための場所"""
    UPPER_VALUE = 19
    """上限値"""

    def __str__(self):
        descriptions = {
            PosId.NO_VALUE: "値無し",
            PosId.MONSTER_L_L: "メインモンスターゾーン左端",
            PosId.MONSTER_L: "メインモンスターゾーン左から2番目",
            PosId.MONSTER_C: "メインモンスターゾーン中央",
            PosId.MONSTER_R: "メインモンスターゾーン右から2番目",
            PosId.MONSTER_R_R: "メインモンスターゾーン右端",
            PosId.EX_L_MONSTER: "EXモンスターゾーン左",
            PosId.EX_R_MONSTER: "EXモンスターゾーン右",
            PosId.MAGIC_L_L: "魔法＆罠ゾーン左端",
            PosId.MAGIC_L: "魔法＆罠ゾーン左から2番目",
            PosId.MAGIC_C: "魔法＆罠ゾーン中央",
            PosId.MAGIC_R: "魔法＆罠ゾーン右から2番目",
            PosId.MAGIC_R_R: "魔法＆罠ゾーン右端",
            PosId.FIELD: "フィールドゾーン",
            PosId.HAND: "手札",
            PosId.EXTRA: "EXデッキ",
            PosId.DECK: "デッキ",
            PosId.GRAVE: "墓地",
            PosId.EXCLUDE: "除外",
            PosId.DISPLAY: "表示のための場所",
            PosId.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class CardIndex(IntEnum):
    """
    その場所にあるカードの何枚目かという値
    """
    NO_VALUE = -1
    """値無し"""

    def __str__(self):
        descriptions = {
            CardIndex.NO_VALUE: "値無し",
        }
        return descriptions.get(self, "不明")

class Frame(IntEnum):
    """
    カードの種類
    """
    NO_VALUE = -1
    """値無し"""
    NORMAL = 0
    """通常"""
    EFFECT = 1
    """効果"""
    FUSION = 2
    """融合"""
    FUSION_FX = 3
    """融合／効果"""
    RITUAL = 4
    """儀式"""
    RITUAL_FX = 5
    """儀式／効果"""
    TOON = 6
    """トゥーン／効果"""
    SPIRIT = 7
    """スピリット／効果"""
    UNION = 8
    """ユニオン／効果"""
    GEMINI = 9
    """デュアル／効果"""
    TOKEN = 10
    """トークン／通常"""
    MAGIC = 13
    """魔法"""
    TRAP = 14
    """罠カード"""
    TUNER = 15
    """チューナー／通常"""
    TUNER_FX = 16
    """チューナー／効果"""
    SYNC = 17
    """シンクロ"""
    SYNC_FX = 18
    """シンクロ／効果"""
    SYNC_TUNER = 19
    """シンクロ／チューナー／効果"""
    XYZ = 22
    """エクシーズ"""
    XYZ_FX = 23
    """エクシーズ／効果"""
    FLIP = 24
    """リバース／効果"""
    PEND = 25
    """ペンデュラム／通常"""
    PEND_FX = 26
    """ペンデュラム／効果"""
    SP_EFFECT = 27
    """特殊召喚／効果"""
    SP_TOON = 28
    """特殊召喚／トゥーン／効果"""
    SP_SPIRIT = 29
    """特殊召喚／スピリット／効果"""
    SP_TUNER = 30
    """特殊召喚／チューナー／効果"""
    FLIP_TUNER = 32
    """リバース／チューナー／効果"""
    PEND_TUNER = 33
    """ペンデュラム／チューナー／効果"""
    XYZ_PEND = 34
    """エクシーズ／ペンデュラム／効果"""
    PEND_FLIP = 35
    """ペンデュラム／リバース／効果"""
    SYNC_PEND = 36
    """シンクロ／ペンデュラム／効果"""
    UNION_TUNER = 37
    """ユニオン／チューナー／効果"""
    RITUAL_SPIRIT = 38
    """儀式／スピリット／効果"""
    FUSION_TUNER = 39
    """融合／チューナー"""
    SP_PEND = 40
    """特殊召喚／ペンデュラム／効果"""
    FUSION_PEND = 41
    """融合／ペンデュラム／効果"""
    LINK = 42
    """リンク"""
    LINK_FX = 43
    """リンク／効果"""
    PEND_N_TUNER = 44
    """ペンデュラム／チューナー／通常"""
    PEND_SPIRIT = 45
    """ペンデュラム／スピリット／効果"""
    RIRUAL_TUNER_FX = 47
    """儀式／チューナー／効果"""
    FUSION_TUNER_FX = 48
    """融合／チューナー／効果"""
    TOKEN_TUNER = 49
    """トークン／チューナー／通常"""
    RITUAL_PEND = 52
    """儀式／ペンデュラム／効果"""
    RITUAL_FLIP = 53
    """儀式／リバース／効果"""

    def __str__(self):
        descriptions = {
            Frame.NO_VALUE: "値無し",
            Frame.NORMAL: "通常",
            Frame.EFFECT: "効果",
            Frame.FUSION: "融合",
            Frame.FUSION_FX: "融合／効果",
            Frame.RITUAL: "儀式",
            Frame.RITUAL_FX: "儀式／効果",
            Frame.TOON: "トゥーン／効果",
            Frame.SPIRIT: "スピリット／効果",
            Frame.UNION: "ユニオン／効果",
            Frame.GEMINI: "デュアル／効果",
            Frame.TOKEN: "トークン／通常",
            Frame.MAGIC: "魔法",
            Frame.TRAP: "罠カード",
            Frame.TUNER: "チューナー／通常",
            Frame.TUNER_FX: "チューナー／効果",
            Frame.SYNC: "シンクロ",
            Frame.SYNC_FX: "シンクロ／効果",
            Frame.SYNC_TUNER: "シンクロ／チューナー／効果",
            Frame.XYZ: "エクシーズ",
            Frame.XYZ_FX: "エクシーズ／効果",
            Frame.FLIP: "リバース／効果",
            Frame.PEND: "ペンデュラム／通常",
            Frame.PEND_FX: "ペンデュラム／効果",
            Frame.SP_EFFECT: "特殊召喚／効果",
            Frame.SP_TOON: "特殊召喚／トゥーン／効果",
            Frame.SP_SPIRIT: "特殊召喚／スピリット／効果",
            Frame.SP_TUNER: "特殊召喚／チューナー／効果",
            Frame.FLIP_TUNER: "リバース／チューナー／効果",
            Frame.PEND_TUNER: "ペンデュラム／チューナー／効果",
            Frame.XYZ_PEND: "エクシーズ／ペンデュラム／効果",
            Frame.PEND_FLIP: "ペンデュラム／リバース／効果",
            Frame.SYNC_PEND: "シンクロ／ペンデュラム／効果",
            Frame.UNION_TUNER: "ユニオン／チューナー／効果",
            Frame.RITUAL_SPIRIT: "儀式／スピリット／効果",
            Frame.FUSION_TUNER: "融合／チューナー",
            Frame.SP_PEND: "特殊召喚／ペンデュラム／効果",
            Frame.FUSION_PEND: "融合／ペンデュラム／効果",
            Frame.LINK: "リンク",
            Frame.LINK_FX: "リンク／効果",
            Frame.PEND_N_TUNER: "ペンデュラム／チューナー／通常",
            Frame.PEND_SPIRIT: "ペンデュラム／スピリット／効果",
            Frame.RIRUAL_TUNER_FX: "儀式／チューナー／効果",
            Frame.FUSION_TUNER_FX: "融合／チューナー／効果",
            Frame.TOKEN_TUNER: "トークン／チューナー／通常",
            Frame.RITUAL_PEND: "儀式／ペンデュラム／効果",
            Frame.RITUAL_FLIP: "儀式／リバース／効果",
        }
        return descriptions.get(self, "不明")

class YesNo(IntEnum):
    """
    YesかNoかの値
    """
    NO_VALUE = -1
    """値無し"""
    NO = 0
    """いいえ"""
    YES = 1
    """はい"""
    UPPER_VALUE = 2
    """上限値"""

    def __str__(self):
        descriptions = {
            YesNo.NO_VALUE: "値無し",
            YesNo.NO: "いいえ",
            YesNo.YES: "はい",
            YesNo.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class EffectNo(IntEnum):
    """
    効果番号
    """
    NO_VALUE = -1
    """値無し"""
    UNKNOWN = 0
    """効果番号不明"""
    CHOICE1 = 1
    """●1"""
    CHOICE2 = 2
    """●2"""
    CHOICE3 = 3
    """●3"""
    CHOICE4 = 4
    """●4"""
    CHOICE5 = 5
    """●5"""
    NUM1 = 11
    """①"""
    NUM2 = 12
    """②"""
    NUM3 = 13
    """③"""
    NUM4 = 14
    """④"""
    NUM5 = 15
    """⑤"""
    PENDULUM1 = 21
    """ペンデュラム①"""
    PENDULUM2 = 22
    """ペンデュラム②"""
    PENDULUM3 = 23
    """ペンデュラム③"""
    PENDULUM4 = 24
    """ペンデュラム④"""
    PENDULUM5 = 25
    """ペンデュラム⑤"""
    UPPER_VALUE = 26
    """上限値"""

    def __str__(self):
        descriptions = {
            EffectNo.NO_VALUE: "値無し",
            EffectNo.UNKNOWN: "効果番号不明",
            EffectNo.CHOICE1: "●1",
            EffectNo.CHOICE2: "●2",
            EffectNo.CHOICE3: "●3",
            EffectNo.CHOICE4: "●4",
            EffectNo.CHOICE5: "●5",
            EffectNo.NUM1: "①",
            EffectNo.NUM2: "②",
            EffectNo.NUM3: "③",
            EffectNo.NUM4: "④",
            EffectNo.NUM5: "⑤",
            EffectNo.PENDULUM1: "ペンデュラム①",
            EffectNo.PENDULUM2: "ペンデュラム②",
            EffectNo.PENDULUM3: "ペンデュラム③",
            EffectNo.PENDULUM4: "ペンデュラム④",
            EffectNo.PENDULUM5: "ペンデュラム⑤",
            EffectNo.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class ChainState(IntEnum):
    """
    チェーンの状態
    """
    NO_VALUE = -1
    """値無し"""
    SET = 0
    """チェーンに積まれる前の処理中"""
    WAIT = 1
    """チェーンに積まれてから効果処理までの間"""
    RESOLVE = 2
    """効果処理中"""
    UPPER_VALUE = 3
    """上限値"""

    def __str__(self):
        descriptions = {
            ChainState.NO_VALUE: "値無し",
            ChainState.SET: "チェーンに積まれる前の処理中",
            ChainState.WAIT: "チェーンに積まれてから効果処理までの間",
            ChainState.RESOLVE: "効果処理中",
            ChainState.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class LockonType(IntEnum):
    """
    CardLockOnの種類
    """
    NO_VALUE = -1
    """値無し"""
    TARGET = 0
    """対象としての選択"""
    NOT_TARGET = 1
    """対象でない選択"""
    CONFIRM = 2
    """相手に見せて確認"""
    ZONE = 3
    """ゾーンを選択"""
    UPPER_VALUE = 4
    """上限値"""

    def __str__(self):
        descriptions = {
            LockonType.NO_VALUE: "値無し",
            LockonType.TARGET: "対象としての選択",
            LockonType.NOT_TARGET: "対象でない選択",
            LockonType.CONFIRM: "相手に見せて確認",
            LockonType.ZONE: "ゾーンを選択",
            LockonType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class DuelLogType(IntEnum):
    """
    デュエルのログの種類
    """
    NO_VALUE = -1
    """値無し"""
    DUEL_START = 0
    """デュエル開始"""
    DUEL_END = 1
    """デュエル終了"""
    BATTLE_ATTACK = 2
    """攻撃宣言"""
    PHASE_CHANGE = 3
    """フェイズ変更"""
    TURN_CHANGE = 4
    """ターン変更"""
    LIFE_SET = 5
    """ライフを設定"""
    LIFE_DAMAGE = 6
    """ライフ増減"""
    HAND_OPEN = 7
    """手札を公開する"""
    DECK_FLIP_TOP = 8
    """デッキの一番上をめくる"""
    CARD_LOCKON = 9
    """対象選択"""
    CARD_MOVE = 10
    """カード移動"""
    CARD_SWAP = 11
    """フィールド上でカード交換"""
    CARD_FLIP_TURN = 12
    """カードの表裏攻守変更"""
    CARD_GENERATE = 13
    """カードの出現"""
    CARD_HAPPEN = 14
    """カード効果発動、効果適用"""
    CARD_DISABLE = 15
    """カード無効"""
    CARD_EQUIP = 16
    """カード装備"""
    CARD_INC_TURN = 17
    """カードターン経過"""
    COUNTER_SET = 18
    """カウンター増減"""
    MONST_SHUFFLE = 19
    """モンスターシャッフル"""
    CHAIN_SET = 20
    """チェーンに積まれる"""
    CHAIN_RUN = 21
    """チェーン解決"""
    RUN_DIALOG = 22
    """ダイアログ表示"""
    RUN_SUMMON = 23
    """召喚"""
    RUN_SP_SUMMON = 24
    """特殊召喚"""
    RUN_COIN = 25
    """コインが投げられた"""
    RUN_DICE = 26
    """サイコロが投げられた"""
    CHAIN_END = 27
    """チェーン処理終了"""
    CHAIN_STEP = 28
    """チェーン処理が有効に開始"""
    RUN_FUSION = 29
    """融合召喚などの特殊召喚の情報"""
    BATTLE_RUN = 30
    """戦闘の情報"""
    UPPER_VALUE = 31
    """上限値"""

    def __str__(self):
        descriptions = {
            DuelLogType.NO_VALUE: "値無し",
            DuelLogType.DUEL_START: "デュエル開始",
            DuelLogType.DUEL_END: "デュエル終了",
            DuelLogType.BATTLE_ATTACK: "攻撃宣言",
            DuelLogType.PHASE_CHANGE: "フェイズ変更",
            DuelLogType.TURN_CHANGE: "ターン変更",
            DuelLogType.LIFE_SET: "ライフを設定",
            DuelLogType.LIFE_DAMAGE: "ライフ増減",
            DuelLogType.HAND_OPEN: "手札を公開する",
            DuelLogType.DECK_FLIP_TOP: "デッキの一番上をめくる",
            DuelLogType.CARD_LOCKON: "対象選択",
            DuelLogType.CARD_MOVE: "カード移動",
            DuelLogType.CARD_SWAP: "フィールド上でカード交換",
            DuelLogType.CARD_FLIP_TURN: "カードの表裏攻守変更",
            DuelLogType.CARD_GENERATE: "カードの出現",
            DuelLogType.CARD_HAPPEN: "カード効果発動、効果適用",
            DuelLogType.CARD_DISABLE: "カード無効",
            DuelLogType.CARD_EQUIP: "カード装備",
            DuelLogType.CARD_INC_TURN: "カードターン経過",
            DuelLogType.COUNTER_SET: "カウンター増減",
            DuelLogType.MONST_SHUFFLE: "モンスターシャッフル",
            DuelLogType.CHAIN_SET: "チェーンに積まれる",
            DuelLogType.CHAIN_RUN: "チェーン解決",
            DuelLogType.RUN_DIALOG: "ダイアログ表示",
            DuelLogType.RUN_SUMMON: "召喚",
            DuelLogType.RUN_SP_SUMMON: "特殊召喚",
            DuelLogType.RUN_COIN: "コインが投げられた",
            DuelLogType.RUN_DICE: "サイコロが投げられた",
            DuelLogType.CHAIN_END: "チェーン処理終了",
            DuelLogType.CHAIN_STEP: "チェーン処理が有効に開始",
            DuelLogType.RUN_FUSION: "融合召喚などの特殊召喚の情報",
            DuelLogType.BATTLE_RUN: "戦闘の情報",
            DuelLogType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class MoveType(IntEnum):
    """
    移動の種類
    """
    NO_VALUE = -1
    """値無し"""
    NORMAL = 0
    """通常移動"""
    SUMMON = 1
    """召喚"""
    SP_SUMMON = 2
    """特殊召喚"""
    ACTIVATE = 3
    """発動"""
    SET = 4
    """セット"""
    BREAK = 5
    """破壊"""
    EXPLOSION = 6
    """戦闘破壊"""
    RELEASE = 7
    """リリース"""
    DRAW = 8
    """ドロー"""
    DROP = 9
    """捨てる"""
    USED = 10
    """使用後"""
    PUT = 11
    """表側で置く"""
    UPPER_VALUE = 12
    """上限値"""

    def __str__(self):
        descriptions = {
            MoveType.NO_VALUE: "値無し",
            MoveType.NORMAL: "通常移動",
            MoveType.SUMMON: "召喚",
            MoveType.SP_SUMMON: "特殊召喚",
            MoveType.ACTIVATE: "発動",
            MoveType.SET: "セット",
            MoveType.BREAK: "破壊",
            MoveType.EXPLOSION: "戦闘破壊",
            MoveType.RELEASE: "リリース",
            MoveType.DRAW: "ドロー",
            MoveType.DROP: "捨てる",
            MoveType.USED: "使用後",
            MoveType.PUT: "表側で置く",
            MoveType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class DamageType(IntEnum):
    """
    ダメージの種類
    """
    NO_VALUE = -1
    """値無し"""
    BY_EFFECT = 0
    """効果ダメージ"""
    BY_BATTLE = 1
    """戦闘ダメージ"""
    BY_COST = 2
    """コストで払う"""
    BY_LOST = 3
    """失う"""
    RECOVER = 4
    """回復"""
    BY_PAY = 5
    """コスト以外で払う"""
    UPPER_VALUE = 6
    """上限値"""

    def __str__(self):
        descriptions = {
            DamageType.NO_VALUE: "値無し",
            DamageType.BY_EFFECT: "効果ダメージ",
            DamageType.BY_BATTLE: "戦闘ダメージ",
            DamageType.BY_COST: "コストで払う",
            DamageType.BY_LOST: "失う",
            DamageType.RECOVER: "回復",
            DamageType.BY_PAY: "コスト以外で払う",
            DamageType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class OpenType(IntEnum):
    """
    手札公開の種類
    """
    NO_VALUE = -1
    """値無し"""
    CLOSE = 0
    """裏にする"""
    OPEN = 1
    """表にする"""
    COST = 2
    """発動コスト・維持コストで見せる"""
    REVEAL = 3
    """処理で『見せる』"""
    UPPER_VALUE = 4
    """上限値"""

    def __str__(self):
        descriptions = {
            OpenType.NO_VALUE: "値無し",
            OpenType.CLOSE: "裏にする",
            OpenType.OPEN: "表にする",
            OpenType.COST: "発動コスト・維持コストで見せる",
            OpenType.REVEAL: "処理で『見せる』",
            OpenType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class SummonType(IntEnum):
    """
    召喚の種類
    """
    NO_VALUE = -1
    """値無し"""
    SUMMON = 0
    """召喚"""
    FLIP_SUMMON = 1
    """反転召喚"""
    GEMINI_SUMMON = 2
    """再度召喚"""
    UPPER_VALUE = 3
    """上限値"""

    def __str__(self):
        descriptions = {
            SummonType.NO_VALUE: "値無し",
            SummonType.SUMMON: "召喚",
            SummonType.FLIP_SUMMON: "反転召喚",
            SummonType.GEMINI_SUMMON: "再度召喚",
            SummonType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class CommandType(IntEnum):
    """
    AIが行う行動の種類
    """
    NO_VALUE = -1
    """値無し"""
    ACTIVATE = 0
    """効果発動"""
    APPLY = 1
    """効果適用"""
    ATTACK = 2
    """攻撃"""
    CHANGE_PHASE = 3
    """フェイズ移行"""
    DECIDE = 4
    """決定"""
    DRAW = 5
    """ドロー"""
    FINALIZE = 6
    """選択終了"""
    PASS = 7
    """発動・適用しない"""
    PENDULUM = 8
    """ペンデュラムスケールに発動"""
    REVERSE = 9
    """反転召喚"""
    SET = 10
    """魔法・罠カードのセット"""
    SET_MONST = 11
    """モンスターのセット"""
    SUMMON = 12
    """召喚"""
    SUMMON_SP = 13
    """特殊召喚"""
    TURN_ATK = 14
    """攻撃表示に変更"""
    TURN_DEF = 15
    """守備表示に変更"""
    UPPER_VALUE = 16
    """上限値"""

    def __str__(self):
        descriptions = {
            CommandType.NO_VALUE: "値無し",
            CommandType.ACTIVATE: "効果発動",
            CommandType.APPLY: "効果適用",
            CommandType.ATTACK: "攻撃",
            CommandType.CHANGE_PHASE: "フェイズ移行",
            CommandType.DECIDE: "決定",
            CommandType.DRAW: "ドロー",
            CommandType.FINALIZE: "選択終了",
            CommandType.PASS: "発動・適用しない",
            CommandType.PENDULUM: "ペンデュラムスケールに発動",
            CommandType.REVERSE: "反転召喚",
            CommandType.SET: "魔法・罠カードのセット",
            CommandType.SET_MONST: "モンスターのセット",
            CommandType.SUMMON: "召喚",
            CommandType.SUMMON_SP: "特殊召喚",
            CommandType.TURN_ATK: "攻撃表示に変更",
            CommandType.TURN_DEF: "守備表示に変更",
            CommandType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class SelectionType(IntEnum):
    """
    入力要求の種類
    """
    NO_VALUE = -1
    """値無し"""
    DRAW_PHASE = 0
    """ドローフェイズ"""
    MAIN_PHASE = 1
    """メインフェイズ"""
    BATTLE_PHASE = 2
    """バトルフェイズ"""
    SELECT_LOCATION = 3
    """場所選択"""
    SELECT_ATTACK_TARGET = 4
    """攻撃対象選択"""
    CHECK_ACTIVATION = 5
    """発動確認"""
    SUMMONING = 6
    """モンスターの召喚中"""
    SP_SUMMONING = 7
    """モンスターの特殊召喚中"""
    SETTING = 8
    """魔法罠のセット中"""
    CHAIN_SETTING = 9
    """チェーンに積まれる前の処理中"""
    CHAIN_RUNNING = 10
    """チェーンの効果処理中"""
    EFFECT_APPLYING = 11
    """割り込みして適用する処理"""
    OTHER = 12
    """その他"""
    UPPER_VALUE = 13
    """上限値"""

    def __str__(self):
        descriptions = {
            SelectionType.NO_VALUE: "値無し",
            SelectionType.DRAW_PHASE: "ドローフェイズ",
            SelectionType.MAIN_PHASE: "メインフェイズ",
            SelectionType.BATTLE_PHASE: "バトルフェイズ",
            SelectionType.SELECT_LOCATION: "場所選択",
            SelectionType.SELECT_ATTACK_TARGET: "攻撃対象選択",
            SelectionType.CHECK_ACTIVATION: "発動確認",
            SelectionType.SUMMONING: "モンスターの召喚中",
            SelectionType.SP_SUMMONING: "モンスターの特殊召喚中",
            SelectionType.SETTING: "魔法罠のセット中",
            SelectionType.CHAIN_SETTING: "チェーンに積まれる前の処理中",
            SelectionType.CHAIN_RUNNING: "チェーンの効果処理中",
            SelectionType.EFFECT_APPLYING: "割り込みして適用する処理",
            SelectionType.OTHER: "その他",
            SelectionType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

class FusionType(IntEnum):
    """
    特殊召喚の種類
    """
    NO_VALUE = -1
    """値無し"""
    FUSION = 0
    """融合"""
    SP_FUSION = 1
    """特殊な融合"""
    SYNCHRO = 2
    """シンクロ"""
    RITUAL = 3
    """儀式"""
    XYZ = 4
    """エクシーズ"""
    PENDULUM = 5
    """ペンデュラム"""
    LINK = 6
    """リンク"""
    UPPER_VALUE = 7
    """上限値"""

    def __str__(self):
        descriptions = {
            FusionType.NO_VALUE: "値無し",
            FusionType.FUSION: "融合",
            FusionType.SP_FUSION: "特殊な融合",
            FusionType.SYNCHRO: "シンクロ",
            FusionType.RITUAL: "儀式",
            FusionType.XYZ: "エクシーズ",
            FusionType.PENDULUM: "ペンデュラム",
            FusionType.LINK: "リンク",
            FusionType.UPPER_VALUE: "上限値",
        }
        return descriptions.get(self, "不明")

