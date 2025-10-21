import copy
import PIL.ImageTk as Itk
import tkinter as tk


from ygo import constants as c
from ygo import models as mdl
from ygo.gui.manager.command_manager import CommandLabel, CommandManager
from ygo.gui.manager.const import Const
from ygo.gui.manager.util import make_command_icon, make_command_text
from ygo.gui.udi_gui_frame import UdiGUIFrame
from ygo.models.command_request import CommandEntry
from ygo.udi_io import UdiIO


class GUILabel(CommandLabel):
    """
    GUIラベル
    """

    def __init__(
        self,
        command_manager: CommandManager,
        master: tk.Misc,
        num: int,
        img: Itk.PhotoImage,
        card: mdl.DuelCard,
        table_index: int,
        text: str,
        subtext: str,
        udi_gui_frame: UdiGUIFrame,
        factor: float,
    ) -> None:
        """
        初期化する。
        """
        tk.Frame.__init__(self, master)

        self.parent_command_manager: CommandManager = command_manager
        self.img: Itk.PhotoImage = img
        self.card: mdl.DuelCard = card
        self.table_index: int = table_index
        self.text: str = text
        self.subtext: str = subtext
        self.ai_text: str = ""
        self.udi_gui_frame: UdiGUIFrame = udi_gui_frame
        self.num: int = num

        scaled_num_font: tuple = (
            Const.COMMAND_NUM_FONT[0],
            int(int(Const.COMMAND_NUM_FONT[1]) * factor),
            Const.COMMAND_NUM_FONT[2],
        )
        scaled_text_font: tuple = (
            Const.COMMAND_TEXT_FONT[0],
            int(int(Const.COMMAND_TEXT_FONT[1]) * factor),
            Const.COMMAND_TEXT_FONT[2],
        )
        scaled_subtext_font: tuple = (
            Const.COMMAND_SUBTEXT_FONT[0],
            int(int(Const.COMMAND_SUBTEXT_FONT[1]) * factor),
        )
        scaled_button_font: tuple = (
            Const.COMMAND_BUTTON_FONT[0],
            int(int(Const.COMMAND_BUTTON_FONT[1]) * factor),
            Const.COMMAND_BUTTON_FONT[2],
        )

        scaled_img_width: int = int(Const.M_CARD_W * factor)
        scaled_img_height: int = int(Const.M_CARD_H * factor)
        scaled_wrap_length: int = int(Const.COMMAND_WRAP_LENGTH * factor)
        scaled_button_padx: int = int(Const.COMMAND_BUTTON_PADX * factor)
        scaled_button_pady: int = int(Const.COMMAND_BUTTON_PADY * factor)

        # コマンド番号
        num_label: tk.Label = tk.Label(self, text=str(num), font=scaled_num_font)
        num_label.pack(side=tk.LEFT)

        # メインのカード
        img_label_dir: tk.Frame = tk.Frame(self, width=scaled_img_width, height=scaled_img_height)
        img_label_dir.propagate(False)
        img_label_dir.pack(side=tk.LEFT)
        img_label: tk.Label = tk.Label(img_label_dir, image=self.img)
        img_label.pack(side=tk.TOP)
        if self.table_index != -1:
            img_label.bind(
                "<Button-1>", lambda event, _table_index=self.table_index: self.call_card_text_manager(_table_index)
            )

        # コマンドのテキスト
        text_dir: tk.Frame = tk.Frame(self)
        text_dir.pack(side=tk.LEFT)
        self.text_label: tk.Label = tk.Label(
            text_dir, text=self.text, font=scaled_text_font, wraplength=scaled_wrap_length
        )
        self.text_label.pack(side=tk.TOP)
        self.subtext_label: tk.Label = tk.Label(
            text_dir, text=self.subtext, font=scaled_subtext_font, wraplength=scaled_wrap_length
        )
        self.subtext_label.pack(side=tk.TOP)
        self.ai_text_label: tk.Label = tk.Label(
            text_dir, text=self.ai_text, font=scaled_subtext_font, wraplength=scaled_wrap_length, foreground="#ff0000"
        )
        self.ai_text_label.pack(side=tk.TOP)

        # コマンド実行ボタン
        self.b_send: tk.Button = tk.Button(self, text="実行", font=scaled_button_font, command=self.send_command)
        self.b_send.pack(side=tk.LEFT, padx=scaled_button_padx, pady=scaled_button_pady)


class GUICommand(CommandManager):
    """
    GUIコマンド
    """

    def update(self, command_request: mdl.CommandRequest, duel_state_data: mdl.DuelStateData) -> None:
        """
        更新する。
        """
        self.reset()

        factor: float = self.udi_gui_frame.factor

        duel_card_table: list[mdl.DuelCard] = duel_state_data.duel_card_table
        commands: list[CommandEntry] = command_request.commands

        # 各コマンドから情報を取得
        for i, command in enumerate(commands):
            ##################################################
            # コマンドに関係するカード
            table_index: int = command.table_index
            card: mdl.DuelCard | None
            card_id: int

            if table_index == -1:
                card = None
                card_id = -1
            else:
                card = duel_card_table[table_index]
                card_id = card.card_id

            # コマンドをテキスト化
            text = ""

            if table_index == -1:
                pass
            else:
                if table_index < 100:
                    text += "(自分)"
                else:
                    text += "(相手)"

                if card_id in (0, -1):
                    text += "裏側カード"
                else:
                    try:
                        text += self.udi_gui_frame.card_util.get_name(card_id)
                    except KeyError:
                        text += "不明カード"

            text += "\n"
            text += make_command_text(command)

            # コマンドをそのまま表示
            subtext: str = str(command)

            ##################################################
            # 画像生成+GUI反映部分
            icon_type, icon_id = make_command_icon(command)

            if icon_type == UdiIO.RatingTextType.CARD_ID or icon_type == UdiIO.RatingTextType.ETC:
                if table_index == -1:
                    # ダミーのカードを作成
                    dummy_dict = {
                        "cardId": 0,
                        "playerId": 0,
                        "posId": -1,
                        "cardIndex": -1,
                        "face": 0,
                        "turn": 0,
                        "isDisabled": -1,
                        "atkVal": -1,
                        "defVal": -1,
                        "isAttacking": -1,
                        "isAttacked": -1,
                        "equipTarget": -1,
                        "magicCounterNum": -1,
                        "usedEffect1": -1,
                        "usedEffect2": -1,
                        "usedEffect3": -1,
                        "turnPassed": -1,
                        "level": -1,
                    }
                    card = mdl.DuelCard(dummy_dict)

                if command.command_type == c.CommandType.SET or command.command_type == c.CommandType.SET_MONST:
                    card = copy.deepcopy(card)

                    if card is not None:
                        card.turn = 1

                img = self.udi_gui_frame.medium_image_manager.get_image_by_card(card)
            else:
                icon_id = int(icon_id)
                img = self.udi_gui_frame.medium_image_manager.get_icon_image(icon_type, icon_id)

            tkimg: Itk.PhotoImage = Itk.PhotoImage(img)  # カード、アイコンの画像

            if card is not None:
                label: GUILabel = GUILabel(
                    self, self.frame, i, tkimg, card, table_index, text, subtext, self.udi_gui_frame, factor
                )
                label.pack(pady=int(Const.COMMAND_PADY * factor), side=tk.TOP, anchor=tk.W)
                self.label_list.append(label)

                for child in label.children.values():
                    child.bind("<MouseWheel>", self._on_mousewheel)

                    for g_child in child.children.values():
                        g_child.bind("<MouseWheel>", self._on_mousewheel)

                label.bind("<MouseWheel>", self._on_mousewheel)
