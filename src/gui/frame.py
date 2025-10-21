from queue import Queue
import tkinter as tk
from typing import Optional, Union

from ygo import constants as c
from ygo.gui.manager.board_manager import BoardManager
from ygo.gui.manager.card_list_manager import CardListManager
from ygo.gui.manager.card_text_manager import CardTextManager
from ygo.gui.manager.chain_manager import ChainManager
from ygo.gui.manager.command_manager import CommandManager
from ygo.gui.manager.const import Const
from ygo.gui.manager.context_manager import ContextManager
from ygo.gui.manager.hand_manager import HandManager
from ygo.gui.manager.image_customizer import ImageCustomizer
from ygo.gui.manager.log_manager import LogManager
from ygo.gui.udi_gui_frame import UdiGUIFrame
from ygo.util.card import CardUtil
from ygo.util.text import TextUtil

from src.gui.manager.dialog import GUIDialog


class GUIFrame(UdiGUIFrame):
    """
    GUIフレーム
    """

    def __init__(self, master: Optional[tk.Misc] = None, queue: Optional[Queue] = None) -> None:
        """
        初期化する。
        """
        tk.Frame.__init__(self, master)

        self.factor: float = 0.5
        self.is_ready: bool = False

        self.small_image_manager: ImageCustomizer = ImageCustomizer(
            int(Const.S_CARD_H * self.factor), int(Const.S_CARD_W * self.factor)
        )
        self.medium_image_manager: ImageCustomizer = ImageCustomizer(
            int(Const.M_CARD_H * self.factor), int(Const.M_CARD_W * self.factor)
        )
        self.large_image_manager: ImageCustomizer = ImageCustomizer(
            int(Const.L_CARD_H * self.factor), int(Const.L_CARD_W * self.factor)
        )
        self.card_util: CardUtil = CardUtil()
        self.text_util: TextUtil = TextUtil()
        self.command_queue: Queue = queue if queue is not None else Queue(1)

        # 記憶部
        self.memories: list = []
        self.memories_list: list = []
        self.memories_info_list: list = []
        self.time: int = -1

        ##################################################
        # GUI設定
        root: Union[tk.Tk, tk.Toplevel] = self.winfo_toplevel()
        root.title("UDI GUI")
        geo_parts = Const.GEO_MAIN.split("x")
        scaled_width = int(int(geo_parts[0]) * self.factor)
        scaled_height = int(int(geo_parts[1]) * self.factor)
        root.geometry(f"{scaled_width}x{scaled_height}")

        # メニューバー
        menu_bar: tk.Menu = tk.Menu(root)
        root.config(menu=menu_bar)
        file_menu: tk.Menu = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="ファイルを読み込む", command=self.open_file)
        file_menu.add_command(label="フォルダを読み込む", command=self.open_folder)
        file_menu.add_command(label="デュエルを指定して再生", command=self.load_duel)

        help_menu: tk.Menu = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="ファイル・フォルダ読み込みについて", command=self.about_file)
        help_menu.add_command(label="コマンド実行について", command=self.about_exec_command)

        # 追加
        additional_dir: tk.Frame = tk.Frame(root, width=int(Const.ADDITIONAL_DIR_WIDTH * self.factor))
        additional_dir.propagate(False)
        additional_dir.pack(side=tk.RIGHT, fill=tk.Y)

        # 右
        right_dir: tk.Frame = tk.Frame(root, width=int(Const.RIGHT_DIR_WIDTH * self.factor))
        right_dir.propagate(False)
        right_dir.pack(side=tk.RIGHT, fill=tk.Y)

        # 中央
        mid_dir: tk.Frame = tk.Frame(root, width=int(Const.MID_DIR_WIDTH * self.factor))
        mid_dir.propagate(False)
        mid_dir.pack(side=tk.RIGHT, fill=tk.Y)

        # 左
        left_dir: tk.Frame = tk.Frame(root)
        left_dir.propagate(False)
        left_dir.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        ##################################################
        # 左

        # ダイアログ
        dialog_dir = tk.LabelFrame(
            left_dir, text="Selection Type, Selection ID", height=int(Const.DIALOG_DIR_HEIGHT * self.factor)
        )
        dialog_dir.propagate(False)
        dialog_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.dialog_manager = GUIDialog(self, dialog_dir)

        # コマンド
        command_dir = tk.LabelFrame(left_dir, text="Commands")
        command_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), expand=True, fill=tk.BOTH)
        self.command_manager = CommandManager(self, command_dir)

        # 巻き戻し・次送り機能
        time_dir = tk.Frame(left_dir)
        time_dir.pack(padx=int(2 * self.factor), pady=int(2 * self.factor))

        scaled_font_size = int(15 * self.factor)
        scaled_font = ("MSゴシック", scaled_font_size, "bold")

        self.b_back = tk.Button(time_dir, text="<", width=10, font=scaled_font, command=self.back)
        self.b_back.pack(side=tk.LEFT, padx=int(2 * self.factor), pady=int(2 * self.factor))
        self.b_back.config(state=tk.DISABLED)
        self.key_back_is_enable = False
        self.master.bind("<Left>", lambda event: self.key_back())

        self.b_pause = tk.Button(time_dir, text="⏸", width=10, font=scaled_font, command=self.pause)
        self.b_pause.pack(side=tk.LEFT, padx=int(2 * self.factor), pady=int(2 * self.factor))
        self.b_pause.config(state=tk.DISABLED)

        self.b_resume = tk.Button(time_dir, text="▶", width=10, font=scaled_font, command=self.resume)
        self.b_resume.pack(side=tk.LEFT, padx=int(2 * self.factor), pady=int(2 * self.factor))
        self.b_resume.config(state=tk.DISABLED)

        self.b_forward = tk.Button(time_dir, text=">", width=10, font=scaled_font, command=self.forward)
        self.b_forward.pack(side=tk.LEFT, padx=int(2 * self.factor), pady=int(2 * self.factor))
        self.b_forward.config(state=tk.DISABLED)
        self.key_forward_is_enable = False
        self.master.bind("<Right>", lambda event: self.key_forward())

        # カードテキスト
        card_text_dir = tk.LabelFrame(left_dir, text="Card Text", height=int(Const.CARD_TEXT_DIR_HEIGHT * self.factor))
        card_text_dir.propagate(False)
        card_text_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.card_text_manager = CardTextManager(self, card_text_dir)

        ##################################################
        # 右

        # チェーンスタック
        chain_dir = tk.LabelFrame(right_dir, text="Chain Stack", height=int(Const.CHAIN_DIR_HEIGHT * self.factor))
        chain_dir.propagate(False)
        chain_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.chain_manager = ChainManager(self, chain_dir)

        # カードリスト
        card_list_dir = tk.LabelFrame(right_dir, text="Card List")
        card_list_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), expand=True, fill=tk.BOTH)
        self.card_list_manager = CardListManager(self, card_list_dir)

        ##################################################
        # 中央

        # p1の手札
        rival_hand_dir = tk.LabelFrame(mid_dir, text="Rival Hand", height=int(Const.HAND_DIR_HEIGHT * self.factor))
        rival_hand_dir.propagate(False)
        rival_hand_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.rival_hand_manager = HandManager(self, c.enums.PlayerId.RIVAL, rival_hand_dir)

        # 盤面
        board_dir = tk.LabelFrame(mid_dir, text="Board")
        board_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), expand=True, fill=tk.BOTH)
        self.board_manager = BoardManager(self, board_dir)

        # p0の手札
        my_hand_dir = tk.LabelFrame(mid_dir, text="My Hand", height=int(Const.HAND_DIR_HEIGHT * self.factor))
        my_hand_dir.propagate(False)
        my_hand_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.my_hand_manager = HandManager(self, c.enums.PlayerId.MYSELF, my_hand_dir)

        ##################################################
        # 追加

        # コンテキスト
        context_dir = tk.LabelFrame(
            additional_dir, text="Command Log", height=int(Const.CONTEXT_DIR_HEIGHT * self.factor)
        )
        context_dir.propagate(False)
        context_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.context_manager = ContextManager(self, context_dir)

        # デュエルログ
        log_dir = tk.LabelFrame(additional_dir, text="Duel Log Data")
        log_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), expand=True, fill=tk.BOTH)
        self.log_manager = LogManager(self, log_dir)

        ##################################################
        self.is_ready = True
