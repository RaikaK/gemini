import os
from queue import Queue
import tkinter as tk
from typing import Optional, Union

from ygo import constants as c
from ygo import models as mdl
from ygo.gui.manager.const import Const
from ygo.gui.manager.hand_manager import HandManager
from ygo.gui.manager.image_customizer import ImageCustomizer
from ygo.gui.udi_gui_frame import UdiGUIFrame
from ygo.util.card import CardUtil
from ygo.util.text import TextUtil

import src.config as config
from src.gui.manager.board import GUIBoard
from src.gui.manager.dialog import GUIDialog
from src.gui.manager.card_list import GUICardList
from src.gui.manager.card_text import GUICardText
from src.gui.manager.chain import GUIChain
from src.gui.manager.command import GUICommand
from src.gui.manager.context import GUIContext
from src.gui.manager.log import GUILog

# デバッグフラグ
DEBUG_MODE = False

# ブレイクポイントフラグ
BREAKPOINT_MODE = False

# ズーム率
DEFAULT_FACTOR = 0.8

# 幅の比率
WIDTH_RATIOS = {
    "left": 0.35,
    "mid": 0.37,
    "right": 0.1,
    "additional": 0.18,
}


class GUIFrame(UdiGUIFrame):
    """
    GUIフレーム
    """

    def __init__(self, master: Optional[tk.Misc] = None, queue: Optional[Queue] = None) -> None:
        """
        初期化する。
        """
        tk.Frame.__init__(self, master)

        self.debug_mode: bool = DEBUG_MODE
        self.breakpoint_mode: bool = BREAKPOINT_MODE
        self.factor: float = DEFAULT_FACTOR
        self.latest_udi_log_data: mdl.UdiLogData | None = None

        self.is_ready: bool = False

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
        self.root: Union[tk.Tk, tk.Toplevel] = self.winfo_toplevel()
        self.root.title("UDI GUI App")

        # メニューバー
        self.menu_bar_frame: tk.Frame = tk.Frame(self.root)
        self.menu_bar_frame.pack(side=tk.TOP, fill=tk.X)
        menu_bar: tk.Menu = tk.Menu(self.menu_bar_frame)
        self.root.config(menu=menu_bar)

        # ズームフレーム
        zoom_frame: tk.Frame = tk.Frame(self.menu_bar_frame)
        zoom_frame.pack(side=tk.RIGHT, padx=int(5 * self.factor))
        zoom_in_button: tk.Button = tk.Button(zoom_frame, text="拡大", command=self._zoom_in, width=8)
        zoom_in_button.pack(side=tk.LEFT)
        zoom_out_button: tk.Button = tk.Button(zoom_frame, text="縮小", command=self._zoom_out, width=8)
        zoom_out_button.pack(side=tk.LEFT)
        self.zoom_label: tk.Label = tk.Label(zoom_frame, text=f"{self.factor:.2f}x", width=6, anchor="e")
        self.zoom_label.pack(side=tk.LEFT, padx=int(5 * self.factor))

        # デバッグフレーム
        self.debug_var = tk.BooleanVar(value=self.debug_mode)
        debug_check_button = tk.Checkbutton(
            self.menu_bar_frame,
            text="デバッグ",
            variable=self.debug_var,
            command=self._toggle_debug,
        )
        debug_check_button.pack(side=tk.RIGHT, padx=int(5 * self.factor))

        # ブレイクポイントフレーム
        self.breakpoint_var = tk.BooleanVar(value=BREAKPOINT_MODE)
        breakpoint_check_button = tk.Checkbutton(
            self.menu_bar_frame,
            text="ブレイクポイント",
            variable=self.breakpoint_var,
            command=self._toggle_breakpoint,
        )
        breakpoint_check_button.pack(side=tk.RIGHT, padx=int(5 * self.factor))

        # 試合数フレーム
        match_frame: tk.Frame = tk.Frame(self.menu_bar_frame)
        match_frame.pack(side=tk.TOP, pady=int(5 * self.factor))
        self.match_label: tk.Label = tk.Label(
            match_frame,
            text="🐉🐉🐉 何試合目 🐉🐉🐉",
            font=("MSゴシック", 10, "bold"),
            width=24,
            anchor="center",
        )
        self.match_label.pack()

        # ファイルメニュー
        file_menu: tk.Menu = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="ファイルを読み込む", command=self.open_file)
        file_menu.add_command(label="フォルダを読み込む", command=self.open_folder)
        file_menu.add_command(label="デュエルを指定して再生", command=self.load_duel)

        # ヘルプメニュー
        help_menu: tk.Menu = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="ファイル・フォルダ読み込みについて", command=self.about_file)
        help_menu.add_command(label="コマンド実行について", command=self.about_exec_command)

        ##################################################
        # メイン領域
        self.main_frame: tk.Frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ##################################################
        # レイアウト更新
        self._update_layout()

        ##################################################
        self.is_ready = True

    def update(self, udi_log_data: mdl.UdiLogData) -> None:  # type: ignore[override]
        """
        表示内容を更新する。
        """
        self.latest_udi_log_data = udi_log_data

        if self.breakpoint_mode:
            breakpoint()

        # 試合数更新
        if os.path.isdir(config.DEMONSTRATION_DIR):
            count = len([f for f in os.listdir(config.DEMONSTRATION_DIR) if f.endswith(".pkl")])
            self.match_label.config(text=f"🐉🐉🐉 {count + 1}試合目 🐉🐉🐉")

        super().update(udi_log_data)

        # ログを一番下までスクロール
        self.log_manager.canvas.update_idletasks()
        self.log_manager.canvas.yview_moveto(1.0)

    def _toggle_debug(self) -> None:
        """
        デバッグモードを切り替える。
        """
        if self.is_ready:
            self.debug_mode = self.debug_var.get()

            if self.latest_udi_log_data is not None:
                self.update(self.latest_udi_log_data)

    def _toggle_breakpoint(self) -> None:
        """
        ブレイクポイントモードを切り替える。
        """
        if self.is_ready:
            self.breakpoint_mode = self.breakpoint_var.get()

    def _zoom_in(self) -> None:
        """
        ズームインする。
        """
        if self.is_ready:
            self.factor = min(2.0, self.factor + 0.05)  # 最大2.0倍
            self._update_layout()

            if self.latest_udi_log_data is not None:
                self.update(self.latest_udi_log_data)

    def _zoom_out(self) -> None:
        """
        ズームアウトする。
        """
        if self.is_ready:
            self.factor = max(0.1, self.factor - 0.05)  # 最小0.1倍
            self._update_layout()

            if self.latest_udi_log_data is not None:
                self.update(self.latest_udi_log_data)

    def _update_layout(self) -> None:
        """
        レイアウトを更新する。
        """
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        self.zoom_label.config(text=f"{self.factor:.2f}x")

        self.small_image_manager: ImageCustomizer = ImageCustomizer(
            int(Const.S_CARD_H * self.factor), int(Const.S_CARD_W * self.factor)
        )
        self.medium_image_manager: ImageCustomizer = ImageCustomizer(
            int(Const.M_CARD_H * self.factor), int(Const.M_CARD_W * self.factor)
        )
        self.large_image_manager: ImageCustomizer = ImageCustomizer(
            int(Const.L_CARD_H * self.factor), int(Const.L_CARD_W * self.factor)
        )

        ##################################################
        # GUI設定
        geo_parts: list = Const.GEO_MAIN.split("x")
        scaled_width: int = int(int(geo_parts[0]) * self.factor)
        scaled_height: int = int(int(geo_parts[1]) * self.factor)
        self.root.geometry(f"{scaled_width}x{scaled_height}")

        # 左
        left_dir: tk.Frame = tk.Frame(self.main_frame, width=int(scaled_width * WIDTH_RATIOS["left"]))
        left_dir.propagate(False)
        left_dir.pack(side=tk.LEFT, fill=tk.Y)

        # 中央
        mid_dir: tk.Frame = tk.Frame(self.main_frame, width=int(scaled_width * WIDTH_RATIOS["mid"]))
        mid_dir.propagate(False)
        mid_dir.pack(side=tk.LEFT, fill=tk.Y)

        # 右
        right_dir: tk.Frame = tk.Frame(self.main_frame, width=int(scaled_width * WIDTH_RATIOS["right"]))
        right_dir.propagate(False)
        right_dir.pack(side=tk.LEFT, fill=tk.Y)

        # 追加
        additional_dir: tk.Frame = tk.Frame(self.main_frame, width=int(scaled_width * WIDTH_RATIOS["additional"]))
        additional_dir.propagate(False)
        additional_dir.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        ##################################################
        # 左

        # ダイアログ
        dialog_dir: tk.LabelFrame = tk.LabelFrame(
            left_dir, text="Selection Type, Selection ID", height=int(Const.DIALOG_DIR_HEIGHT * self.factor)
        )
        dialog_dir.propagate(False)
        dialog_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.dialog_manager: GUIDialog = GUIDialog(self, dialog_dir)

        # コマンド
        command_dir: tk.LabelFrame = tk.LabelFrame(left_dir, text="Commands")
        command_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), expand=True, fill=tk.BOTH)
        self.command_manager: GUICommand = GUICommand(self, command_dir)

        # 巻き戻し・次送り機能
        time_dir: tk.Frame = tk.Frame(left_dir)
        time_dir.pack(padx=int(2 * self.factor), pady=int(2 * self.factor))

        scaled_font_size: int = int(15 * self.factor)
        scaled_font: tuple = ("MSゴシック", scaled_font_size, "bold")

        self.b_back: tk.Button = tk.Button(time_dir, text="<", width=10, font=scaled_font, command=self.back)
        self.b_back.pack(side=tk.LEFT, padx=int(2 * self.factor), pady=int(2 * self.factor))
        self.b_back.config(state=tk.DISABLED)
        self.key_back_is_enable: bool = False
        self.master.bind("<Left>", lambda event: self.key_back())

        self.b_pause: tk.Button = tk.Button(time_dir, text="⏸", width=10, font=scaled_font, command=self.pause)
        self.b_pause.pack(side=tk.LEFT, padx=int(2 * self.factor), pady=int(2 * self.factor))
        self.b_pause.config(state=tk.DISABLED)

        self.b_resume: tk.Button = tk.Button(time_dir, text="▶", width=10, font=scaled_font, command=self.resume)
        self.b_resume.pack(side=tk.LEFT, padx=int(2 * self.factor), pady=int(2 * self.factor))
        self.b_resume.config(state=tk.DISABLED)

        self.b_forward: tk.Button = tk.Button(time_dir, text=">", width=10, font=scaled_font, command=self.forward)
        self.b_forward.pack(side=tk.LEFT, padx=int(2 * self.factor), pady=int(2 * self.factor))
        self.b_forward.config(state=tk.DISABLED)
        self.key_forward_is_enable: bool = False
        self.master.bind("<Right>", lambda event: self.key_forward())

        # カードテキスト
        card_text_dir: tk.LabelFrame = tk.LabelFrame(
            left_dir, text="Card Text", height=int(Const.CARD_TEXT_DIR_HEIGHT * self.factor)
        )
        card_text_dir.propagate(False)
        card_text_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.card_text_manager: GUICardText = GUICardText(self, card_text_dir)

        ##################################################
        # 右

        # チェーンスタック
        chain_dir: tk.LabelFrame = tk.LabelFrame(
            right_dir, text="Chain Stack", height=int(Const.CHAIN_DIR_HEIGHT * self.factor)
        )
        chain_dir.propagate(False)
        chain_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.chain_manager: GUIChain = GUIChain(self, chain_dir)

        # カードリスト
        card_list_dir: tk.LabelFrame = tk.LabelFrame(right_dir, text="Card List")
        card_list_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), expand=True, fill=tk.BOTH)
        self.card_list_manager: GUICardList = GUICardList(self, card_list_dir)

        ##################################################
        # 中央

        # p1の手札
        rival_hand_dir: tk.LabelFrame = tk.LabelFrame(
            mid_dir, text="Rival Hand", height=int(Const.HAND_DIR_HEIGHT * self.factor)
        )
        rival_hand_dir.propagate(False)
        rival_hand_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.rival_hand_manager: HandManager = HandManager(self, c.enums.PlayerId.RIVAL, rival_hand_dir)

        # 盤面
        board_dir: tk.LabelFrame = tk.LabelFrame(mid_dir, text="Board")
        board_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), expand=True, fill=tk.BOTH)
        self.board_manager: GUIBoard = GUIBoard(self, board_dir)

        # p0の手札
        my_hand_dir: tk.LabelFrame = tk.LabelFrame(
            mid_dir, text="My Hand", height=int(Const.HAND_DIR_HEIGHT * self.factor)
        )
        my_hand_dir.propagate(False)
        my_hand_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.my_hand_manager: HandManager = HandManager(self, c.enums.PlayerId.MYSELF, my_hand_dir)

        ##################################################
        # 追加

        # コンテキスト
        context_dir: tk.LabelFrame = tk.LabelFrame(
            additional_dir, text="Command Log", height=int(Const.CONTEXT_DIR_HEIGHT * self.factor)
        )
        context_dir.propagate(False)
        context_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), fill=tk.X)
        self.context_manager: GUIContext = GUIContext(self, context_dir)

        # デュエルログ
        log_dir: tk.LabelFrame = tk.LabelFrame(additional_dir, text="Duel Log Data")
        log_dir.pack(anchor=tk.W, padx=int(2 * self.factor), pady=int(2 * self.factor), expand=True, fill=tk.BOTH)
        self.log_manager: GUILog = GUILog(self, log_dir)
