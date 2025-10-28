import PIL.ImageTk as Itk
import tkinter as tk


from ygo import constants as c
from ygo import models as mdl
from ygo.gui.manager.chain_manager import ChainLabel, ChainManager
from ygo.gui.manager.const import Const


class GUIChainLabel(ChainLabel):
    """
    GUIチェーンラベル
    """

    def __init__(
        self,
        master: tk.Misc,
        num: int,
        img: Itk.PhotoImage,
        text: str,
        card: mdl.DuelCard,
        table_index: int,
        udi_gui_frame,
        factor: float,
    ) -> None:
        """
        初期化する。
        """
        super().__init__(master, num, img, text, card, table_index, udi_gui_frame)

        scaled_bd: int = max(1, int(Const.CHAIN_BD * factor))
        scaled_num_font: tuple = (
            Const.CHAIN_NUM_FONT[0],
            max(8, int(int(Const.CHAIN_NUM_FONT[1]) * factor)),
            Const.CHAIN_NUM_FONT[2],
        )
        scaled_text_font: tuple = (
            Const.CHAIN_TEXT_FONT[0],
            max(8, int(int(Const.CHAIN_TEXT_FONT[1]) * factor)),
        )
        scaled_wrap_length: int = int(Const.CHAIN_WRAP_LENGTH * factor)

        self.config(bd=scaled_bd)
        self.num_label.config(font=scaled_num_font)
        self.text_label.config(font=scaled_text_font, wraplength=scaled_wrap_length)


class GUIChain(ChainManager):
    """
    GUIチェーン
    """

    def update(self, duel_state_data: mdl.DuelStateData) -> None:
        """
        更新する。
        """
        self.reset()

        factor: float = self.udi_gui_frame.factor

        # 各チェーンから情報を取得
        duel_card_table: list[mdl.DuelCard] = duel_state_data.duel_card_table
        chain_stack: list[mdl.ChainData] = duel_state_data.chain_stack

        for i, chain in enumerate(chain_stack):
            text: str = ""

            # 効果を発動したカード
            table_index: int = chain.table_index
            card: mdl.DuelCard = duel_card_table[table_index]

            if table_index < 100:
                text += "(自分)"
            else:
                text += "(相手)"

            # 効果番号
            effect_no: int = chain.effect_no
            text += f"{c.enums.EffectNo(effect_no)}"
            text += "\n"

            # チェーンの状態
            chain_state: int = chain.chain_state
            text += f"{c.enums.ChainState(chain_state)}"
            text += "\n"

            # 効果の対象
            target_table_index_list: list = chain.target_table_index_list

            if len(target_table_index_list) > 0:
                text += "対象："
            for target_table_index in target_table_index_list:
                if target_table_index < 100:
                    text += "(自分)"
                else:
                    text += "(相手)"

                target_card: mdl.DuelCard = duel_card_table[target_table_index]
                target_card_id: int = target_card.card_id

                if target_card_id in (0, -1):
                    text += "裏側カード "
                else:
                    try:
                        text += self.udi_gui_frame.card_util.get_name(target_card_id)
                    except KeyError:
                        text += "不明 "
                    text += " "

            ##################################################
            # 画像生成+GUI反映部分
            img = self.udi_gui_frame.medium_image_manager.get_image_by_card(card)
            tkimg: Itk.PhotoImage = Itk.PhotoImage(img)

            label: GUIChainLabel = GUIChainLabel(
                self.frame, i, tkimg, text, card, table_index, self.udi_gui_frame, factor
            )
            label.pack()

            for child in label.children.values():
                child.bind("<MouseWheel>", self._on_mousewheel)

            label.bind("<MouseWheel>", self._on_mousewheel)
