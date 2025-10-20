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
from ygo.gui.manager.dialog_manager import DialogManager
from ygo.gui.manager.hand_manager import HandManager
from ygo.gui.manager.image_customizer import ImageCustomizer
from ygo.gui.manager.log_manager import LogManager
from ygo.gui.udi_gui_frame import UdiGUIFrame
from ygo.util.card import CardUtil
from ygo.util.text import TextUtil


class GUIFrame(UdiGUIFrame):
    """
    GUIフレーム
    """

    def __init__(self, master: Optional[tk.Misc] = None, queue: Optional[Queue] = None):
        """
        初期化する。
        """
        tk.Frame.__init__(self, master)

        self.is_ready: bool = False

        self.small_image_manager: ImageCustomizer = ImageCustomizer(Const.S_CARD_H, Const.S_CARD_W)
        self.medium_image_manager: ImageCustomizer = ImageCustomizer(Const.M_CARD_H, Const.M_CARD_W)
        self.large_image_manager: ImageCustomizer = ImageCustomizer(Const.L_CARD_H, Const.L_CARD_W)
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
        root.title("UDI GUI App")
        root.columnconfigure(0, weight=2)
        root.columnconfigure(1, weight=5)
        root.columnconfigure(2, weight=2)
        root.columnconfigure(3, weight=2)
        root.rowconfigure(0, weight=1)

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
        self.additional_dir: tk.Frame = tk.Frame(root)
        self.additional_dir.grid(row=0, column=3, sticky="nsew", padx=2, pady=2)

        # 右
        self.right_dir: tk.Frame = tk.Frame(root)
        self.right_dir.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)

        # 中央
        self.mid_dir: tk.Frame = tk.Frame(root)
        self.mid_dir.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

        # 左
        self.left_dir: tk.Frame = tk.Frame(root)
        self.left_dir.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        ##################################################
        # 左
        self.left_dir.rowconfigure(0, weight=0, minsize=Const.DIALOG_DIR_HEIGHT)
        self.left_dir.rowconfigure(1, weight=1)
        self.left_dir.rowconfigure(2, weight=0)
        self.left_dir.rowconfigure(3, weight=0, minsize=Const.CARD_TEXT_DIR_HEIGHT)
        self.left_dir.columnconfigure(0, weight=1)

        # ダイアログ
        self.dialog_dir: tk.LabelFrame = tk.LabelFrame(self.left_dir, text="Selection Type, Selection ID")
        self.dialog_manager: DialogManager = DialogManager(self, self.dialog_dir)
        self.dialog_dir.grid(row=0, column=0, sticky="new", padx=2, pady=2)

        # コマンド
        self.command_dir: tk.LabelFrame = tk.LabelFrame(self.left_dir, text="Commands")
        self.command_manager: CommandManager = CommandManager(self, self.command_dir)
        self.command_dir.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # 巻き戻し・次送り機能
        self.time_dir: tk.Frame = tk.Frame(self.left_dir)
        self.time_dir.grid(row=2, column=0, sticky="ew", padx=2, pady=2)

        self.b_back: tk.Button = tk.Button(
            self.time_dir, text="<", width=10, font=("MS Gothic", 10, "bold"), command=self.back
        )
        self.b_back.pack(side=tk.LEFT, padx=2, pady=2)
        self.b_back.config(state=tk.DISABLED)
        self.key_back_is_enable: bool = False
        root.bind("<Left>", lambda event: self.key_back())

        self.b_pause: tk.Button = tk.Button(
            self.time_dir, text="⏸", width=10, font=("MS Gothic", 10, "bold"), command=self.pause
        )
        self.b_pause.pack(side=tk.LEFT, padx=2, pady=2)
        self.b_pause.config(state=tk.DISABLED)

        self.b_resume: tk.Button = tk.Button(
            self.time_dir, text="▶", width=10, font=("MS Gothic", 10, "bold"), command=self.resume
        )
        self.b_resume.pack(side=tk.LEFT, padx=2, pady=2)
        self.b_resume.config(state=tk.DISABLED)

        self.b_forward: tk.Button = tk.Button(
            self.time_dir, text=">", width=10, font=("MS Gothic", 10, "bold"), command=self.forward
        )
        self.b_forward.pack(side=tk.LEFT, padx=2, pady=2)
        self.b_forward.config(state=tk.DISABLED)
        self.key_forward_is_enable: bool = False
        root.bind("<Right>", lambda event: self.key_forward())

        # カードテキスト
        self.card_text_dir: tk.LabelFrame = tk.LabelFrame(self.left_dir, text="Card Text")
        self.card_text_manager: CardTextManager = CardTextManager(self, self.card_text_dir)
        self.card_text_dir.grid(row=3, column=0, sticky="sew", padx=2, pady=2)

        ##################################################
        # 右
        self.right_dir.rowconfigure(0, weight=0, minsize=Const.CHAIN_DIR_HEIGHT)
        self.right_dir.rowconfigure(1, weight=1)
        self.right_dir.columnconfigure(0, weight=1)

        # チェーンスタック
        self.chain_dir: tk.LabelFrame = tk.LabelFrame(self.right_dir, text="Chain Stack")
        self.chain_manager: ChainManager = ChainManager(self, self.chain_dir)
        self.chain_dir.grid(row=0, column=0, sticky="new", padx=2, pady=2)

        # カードリスト
        self.card_list_dir: tk.LabelFrame = tk.LabelFrame(self.right_dir, text="Card List")
        self.card_list_manager: CardListManager = CardListManager(self, self.card_list_dir)
        self.card_list_dir.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        ##################################################
        # 中央
        self.mid_dir.rowconfigure(0, weight=0, minsize=Const.HAND_DIR_HEIGHT)
        self.mid_dir.rowconfigure(1, weight=1)
        self.mid_dir.rowconfigure(2, weight=0, minsize=Const.HAND_DIR_HEIGHT)
        self.mid_dir.columnconfigure(0, weight=1)

        # p1の手札
        self.rival_hand_dir: tk.LabelFrame = tk.LabelFrame(self.mid_dir, text="Rival Hand")
        self.rival_hand_manager: HandManager = HandManager(self, c.enums.PlayerId.RIVAL, self.rival_hand_dir)
        self.rival_hand_dir.grid(row=0, column=0, sticky="new")

        # 盤面
        self.board_dir: tk.LabelFrame = tk.LabelFrame(self.mid_dir, text="Board")
        self.board_manager: BoardManager = BoardManager(self, self.board_dir)
        self.board_dir.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # p0の手札
        self.my_hand_dir: tk.LabelFrame = tk.LabelFrame(self.mid_dir, text="My Hand")
        self.my_hand_manager: HandManager = HandManager(self, c.enums.PlayerId.MYSELF, self.my_hand_dir)
        self.my_hand_dir.grid(row=2, column=0, sticky="sew")

        ##################################################
        # 追加
        self.additional_dir.rowconfigure(0, weight=0, minsize=Const.CONTEXT_DIR_HEIGHT)
        self.additional_dir.rowconfigure(1, weight=1)
        self.additional_dir.columnconfigure(0, weight=1)

        # コンテキスト
        self.context_dir: tk.LabelFrame = tk.LabelFrame(self.additional_dir, text="Command Log")
        self.context_manager: ContextManager = ContextManager(self, self.context_dir)
        self.context_dir.grid(row=0, column=0, sticky="new", padx=2, pady=2)

        # デュエルログ
        self.log_dir: tk.LabelFrame = tk.LabelFrame(self.additional_dir, text="Duel Log Data")
        self.log_manager: LogManager = LogManager(self, self.log_dir)
        self.log_dir.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        ##################################################
        self.is_ready = True
