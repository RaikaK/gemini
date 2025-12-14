#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.
"""GUIで使用するカード画像を読み込むクラスの定義"""

import os
import json
import io

from PIL import Image

class ImageManager:
    """
    GUIで使用するカード画像を読み込むクラス
    """

    card_h = 50
    card_w = 50
    padding = 10

    text_h = 100

    img_file_name = 'combined_image.bytes'
    map_file_name = 'combined_image_map.json'

    def __init__(self, path = f"{os.path.dirname(__file__)}"):
        with open(f"{os.path.join(path, ImageManager.img_file_name)}", "rb") as f_img_bytes:
            self._img_bytes = f_img_bytes.read()
        with open(f"{os.path.join(path, ImageManager.map_file_name)}", "r") as f_img_map:
            self._img_map = json.loads(f_img_map.read())

    @staticmethod
    def _get_card_image_name(card_id):        
        file_name = f"{card_id}.jpg"
        return file_name

    @staticmethod
    def _resize_image(img: Image.Image):
        w = img.width # 横幅を取得                                            
        h = img.height # 縦幅を取得   
        img = img.resize((int(w * ImageManager.card_h / h), int(h * ImageManager.card_w / h)))
        return img
    
    def _get_image(self, img_name):
        [img_begin, img_length] = self._img_map[img_name]
        img = Image.open(io.BytesIO(self._img_bytes[img_begin:img_begin + img_length]))
        return img
    
    def is_valid_card_id(self, card_id):
        file_name = ImageManager._get_card_image_name(card_id)
        if file_name in self._img_map:
            return True
        else:
            return False

    def get_card_image(self, card_id):
        if not ImageManager.is_valid_card_id(self, card_id):
            return self.get_protector_image()  
        name = ImageManager._get_card_image_name(card_id)
        img = self._get_image(name)
        return img
    
    def get_turned_card_image(self, card_id): 
        if not self.is_valid_card_id(card_id):
            return self.get_protector_image()  
        name = ImageManager._get_card_image_name(card_id)
        img = self._get_image(name)
        img_rotated = img.rotate(90, expand=True)
        return img_rotated
    
    def get_bg_image(self):
        bg_name = "bg.png"
        img = self._get_image(bg_name)
        return img
    
    def get_protector_image(self):
        protector_name = "protector.png"
        img = self._get_image(protector_name)
        return img
    
    def get_phase_image(self, phase):
        name = f"phase_{phase}.png"
        img = self._get_image(name)
        return img
    
    def get_draw_image(self):
        name = "command_draw.png"
        img = self._get_image(name)
        return img
    
    def get_position_image(self):
        name = "position.png"
        img = self._get_image(name)
        return img
    
    def get_coin_image(self, face):
        if face:
            face_name = "head"
        else:
            face_name = "tail"
        name = f"coin_{face_name}.png"
        img = self._get_image(name)
        return img
    
    def get_cancel_image(self):
        name = f"cancel.png"
        img = self._get_image(name)
        return img