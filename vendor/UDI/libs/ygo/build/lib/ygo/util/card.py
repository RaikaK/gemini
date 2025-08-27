#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""カード情報に関するUtil"""

import csv
import os

from ..constants.enums import Attribute, Species, Frame, Icon


class CardUtil:
    """
    カードに記載されている情報を取得するためのクラス
    """

    def __init__(self):
        card_data_path = os.path.join(os.path.dirname(__file__), "../data/card_data.csv")
        self.card_data = {}
        
        with open(card_data_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                card_id = int(row["card_id"])
                self.card_data[card_id] = row
        
    def get_name(self, card_id: int) -> str:
        """
        日本語のカード名を取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            カード名
        """
        return self.card_data[card_id]['name']

    def get_sort_name(self, card_id: int) -> str:
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
        return self.card_data[card_id]['sort_name']

    def get_text(self, card_id) -> str:
        """
        カードテキストを取得する

        Parameters
        ----------
        card_id : card_id
            card_id

        Returns
        -------
        str
            カードテキスト
        """
        return self.card_data[card_id]['text']

    def get_effect_text(self, effect_card_id: int, effect_no: int) -> str:
        """
        効果番号（EffectNo）に対応する効果テキストを取得する

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
        return self.card_data[effect_card_id][str(int(effect_no))]
    
    def get_attribute(self, card_id: int) -> Attribute:
        """
        カードに記載された属性を取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        Attribute
            属性
        """
        return Attribute(int(self.card_data[card_id]["attribute"]))
    
    def get_level(self, card_id: int) -> int:
        """
        カードに記載されたレベルを取得する
        魔法・罠カードは0

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        int
            レベル
        """
        return int(self.card_data[card_id]["level"])
    
    def get_species(self, card_id: int) -> Species:
        """
        カードに記載された種族を取得する
        魔法・罠カードはSpecies.NULL

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        Species
            種族
        """
        return Species(int(self.card_data[card_id]["species"]))
    
    def get_atk(self, card_id: int) -> int:
        """
        カードに記載された攻撃力を取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        int
            カードに記載されている攻撃力（魔法・罠カードは0。?は-1）
        """
        return int(self.card_data[card_id]["atk"])
    
    def get_def(self, card_id: int) -> int:
        """
        カードに記載された守備力を取得する
        
        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        int
            カードに記載されている守備力（魔法・罠カードは0。?は-1）
        """
        return int(self.card_data[card_id]["def"])

    
    def get_frame(self, card_id: int) -> Frame:
        """
        カードの種類（【】内の種族以外の項目）を取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        Frame
            カードの種類（【】内の種族以外の項目）
        """
        return Frame(int(self.card_data[card_id]["frame"]))

    def get_icon(self, card_id: int) -> Icon:
        """
        永続や速攻など魔法・罠カードの種類を表す値を取得する

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        str
            魔法・罠カードの種類
        """
        if self.is_magic_trap(card_id):
            return Icon(int(self.card_data[card_id]["icon"]))
        else:
            return Icon.NO_VALUE

    def get_scale(self, card_id: int) -> int:
        """
        ペンデュラムスケールを取得する
        
        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        int
            ペンデュラムスケール。ペンデュラムモンスター以外の場合は-1が返る
        """
        if self.is_pendulum(card_id):
            return int(self.card_data[card_id]["scale"])
        else:
            return -1
    
    def is_pendulum(self, card_id: int) -> bool:
        """
        ペンデュラムモンスターカードかどうかを返す

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        bool
            ペンデュラムモンスターかどうか
        """
        pend_frame = {
            Frame.PEND,
            Frame.PEND_FX,
            Frame.PEND_TUNER,
            Frame.XYZ_PEND,
            Frame.PEND_FLIP,
            Frame.SYNC_PEND,
            Frame.SP_PEND,
            Frame.FUSION_PEND,
            Frame.PEND_N_TUNER,
            Frame.PEND_SPIRIT,
            Frame.RITUAL_PEND }
        frame = self.get_frame(card_id)
        return frame in pend_frame

    def is_magic_trap(self, card_id: int) -> bool:
        """
        魔法・罠カードかどうかを返す

        Parameters
        ----------
        card_id : int
            card_id

        Returns
        -------
        bool
            魔法・罠カードかどうか
        """
        magic_trap_frame = {
            Frame.MAGIC,
            Frame.TRAP
        }
        frame = self.get_frame(card_id)
        return frame in magic_trap_frame