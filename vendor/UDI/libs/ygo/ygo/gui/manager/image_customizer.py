#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

from PIL import Image, ImageTk, ImageDraw

from ygo.udi_io import UdiIO
from ygo import models as mdl
from ygo.image.image_manager import ImageManager


class ImageCustomizer:
    def __init__(self, h, w):
        self.image_manager = ImageManager()

        self.card_h = h
        self.card_w = w

        # カード画像関連
        self.img_pool = {}
        self.img_pool_turn = {}
        self.img_protector = None
        self.img_protector_turn = None

    def _resize_image(self, img: Image):
        w = img.width # 横幅を取得                                            
        h = img.height # 縦幅を取得

        if h > w:
            longer = h
        else:
            longer = w

        img = img.resize((int(w * self.card_h / longer), int(h * self.card_w / longer)))
        return img
    
    
    def get_card_image(self, card_id) -> ImageTk: 
        if card_id in self.img_pool:
            return self.img_pool[card_id]
        
        img = self.image_manager.get_card_image(card_id)
        img_resized = self._resize_image(img)
        self.img_pool[card_id] = img_resized
        return self.img_pool[card_id]


    def get_turned_card_image(self, card_id) -> ImageTk: 
        if card_id in self.img_pool_turn:
            return self.img_pool_turn[card_id]
        
        img = self.image_manager.get_turned_card_image(card_id)
        img_resized = self._resize_image(img)
        self.img_pool_turn[card_id] = img_resized
        return self.img_pool_turn[card_id]
    
    
    def get_protector_image(self):
        if self.img_protector != None:
            return self.img_protector
        img_resized = self._resize_image(self.image_manager.get_protector_image())
        self.img_protector = img_resized
        return self.img_protector
    
    def get_turned_protector_image(self):
        if self.img_protector_turn != None:
            return self.img_protector_turn
        img_resized = self._resize_image(self.image_manager.get_protector_image()).rotate(90, expand=True)
        self.img_protector_turn = img_resized
        return self.img_protector_turn

    def get_image_by_card(self, card : mdl.DuelCard):
        card_id = card.card_id
        face = card.face
        turn = card.turn
        player = card.player_id

        if face == 1:
            if turn == 1:
                img = self.get_turned_card_image(card_id)
            else:
                img = self.get_card_image(card_id)
        else:
            if turn == 1:
                img = self.get_turned_protector_image()
                if card_id > 0:
                    card_img = self.get_turned_card_image(card_id).convert('RGBA')
            else:
                img = self.get_protector_image()
                if card_id > 0:
                    card_img = self.get_card_image(card_id).convert('RGBA')

            if card_id > 0:
                w, h = card_img.size
                img = img.resize((w,h))
                img = img.convert('RGBA')
                img.putalpha(100)
                img = Image.alpha_composite(card_img, img)

        card_width, card_height = img.size
        frame_img = Image.new("RGBA", (card_width, card_height))
        draw = ImageDraw.Draw(frame_img)
        if card_height > card_width:
            longer = card_height
        else:
            longer = card_width
        if player == 0:
            color  = "#005AFF"
        elif player == 1:
            color = "#FF4B00"
        else:
            color = "#03AF7A"
        draw.rectangle((0, 0, card_width -1, card_height-1), outline = color, width = int(longer/30))
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, frame_img)
        
        return img

    def get_icon_image(self, icon_type, icon_id):
        if icon_type == UdiIO.RatingTextType.PHASE:
            img = self.image_manager.get_phase_image(icon_id)
        elif icon_type == UdiIO.RatingTextType.DRAW:
            img = self.image_manager.get_draw_image()
        elif icon_type == UdiIO.RatingTextType.POSITION:
            img = self.image_manager.get_position_image()
        elif icon_type == UdiIO.RatingTextType.COIN:
            img = self.image_manager.get_coin_image(icon_id)
        elif icon_type == UdiIO.RatingTextType.CANCEL:
            img = self.image_manager.get_cancel_image()

        img_resized = self._resize_image(img)
        return img_resized