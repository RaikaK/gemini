#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import copy
import os
from queue import Queue
import tkinter as tk
import tkinter.filedialog

from ygo import constants as c
from ygo import models as mdl
from ygo.util.text import TextUtil
from ygo.util.card import CardUtil
from ygo.util.udi_log import UdiLogUtil

from .manager.const import Const
from .manager.image_customizer import ImageCustomizer
from .manager.chain_manager import ChainManager
from .manager.card_text_manager import CardTextManager
from .manager.dialog_manager import DialogManager
from .manager.command_manager import CommandManager
from .manager.hand_manager import HandManager
from .manager.board_manager import BoardManager
from .manager.card_list_manager import CardListManager
from .manager.context_manager import ContextManager
from .manager.log_manager import LogManager


# TODO: 現在は試合の区切りが無い 別ウィンドウでロードする試合を切り替えられるといいかも？
class UdiGUIFrame(tk.Frame):
    def __init__(self, master=None, queue = None):
        tk.Frame.__init__(self, master)

        self.is_ready = False

        self.small_image_manager  = ImageCustomizer(Const.S_CARD_H, Const.S_CARD_W)
        self.medium_image_manager = ImageCustomizer(Const.M_CARD_H, Const.M_CARD_W)
        self.large_image_manager  = ImageCustomizer(Const.L_CARD_H, Const.L_CARD_W)
        self.card_util = CardUtil()
        self.text_util = TextUtil()
        self.command_queue : Queue = queue

        # 記憶部
        self.memories = [] # 盤面単位でUdiLogData+αを保存
        self.memories_list = [] # デュエル単位でmemoriesを保存
        self.memories_info_list = [] # デュエル単位で保存したmemoriesの概要を保存
        self.time = -1

        ################################################################################################################
        # gui設定
        self.master.title('UDI GUI App(β)')
        self.master.geometry(Const.GEO_MAIN)
        
        # メニューバー
        menu_bar = tk.Menu(self.master)
        self.master.config(menu=menu_bar)
        file_menu = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label='ファイル', menu=file_menu)
        file_menu.add_command(label='ファイルを読み込む', command=self.open_file)
        file_menu.add_command(label='フォルダを読み込む', command=self.open_folder)
        file_menu.add_command(label='デュエルを指定して再生', command=self.load_duel)

        help_menu = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label='ヘルプ', menu=help_menu)
        help_menu.add_command(label='ファイル・フォルダ読み込みについて', command=self.about_file)
        help_menu.add_command(label='コマンド実行について', command=self.about_exec_command)

        # 追加
        additional_dir = tk.Frame(self.master, width=Const.ADDITIONAL_DIR_WIDTH)
        additional_dir.propagate(False)
        additional_dir.pack(side=tk.RIGHT, fill=tk.Y)

        # 右
        right_dir = tk.Frame(self.master, width=Const.RIGHT_DIR_WIDTH)
        right_dir.propagate(False)
        right_dir.pack(side=tk.RIGHT, fill=tk.Y)

        # 中央
        mid_dir = tk.Frame(self.master, width=Const.MID_DIR_WIDTH)
        mid_dir.propagate(False)
        mid_dir.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 左
        left_dir = tk.Frame(self.master)
        left_dir.propagate(False)
        left_dir.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)


        ################################################################################################################
        # 左（カードテキストを先に定義しておかなくちゃいけない）

        # ダイアログ
        dialog_dir = tk.LabelFrame(left_dir, text='Selection Type, Selection ID', height=Const.DIALOG_DIR_HEIGHT)
        dialog_dir.propagate(False)
        dialog_dir.pack(anchor=tk.W, padx=2, pady=2, fill=tk.X)
        self.dialog_manager = DialogManager(self, dialog_dir)

        # コマンド
        command_dir = tk.LabelFrame(left_dir, text='Commands')
        command_dir.pack(anchor=tk.W, padx=2, pady=2, expand=True, fill=tk.BOTH)
        self.command_manager = CommandManager(self, command_dir)

        # 巻き戻し・次送り機能
        time_dir = tk.Frame(left_dir)
        time_dir.pack(padx=2, pady=2)

        self.b_back=tk.Button(time_dir, text="<", width=10, font=('MSゴシック', '15', "bold"), command=self.back)
        self.b_back.pack(side = tk.LEFT, padx=2, pady=2)
        self.b_back.config(state=tk.DISABLED)
        self.key_back_is_enable = False
        self.master.bind("<Left>", lambda event:self.key_back())

        self.b_pause=tk.Button(time_dir, text="⏸", width=10, font=('MSゴシック', '15', "bold"), command=self.pause)
        self.b_pause.pack(side = tk.LEFT, padx=2, pady=2)
        self.b_pause.config(state=tk.DISABLED)

        self.b_resume=tk.Button(time_dir, text="▶", width=10, font=('MSゴシック', '15', "bold"), command=self.resume)
        self.b_resume.pack(side = tk.LEFT, padx=2, pady=2)
        self.b_resume.config(state=tk.DISABLED)

        self.b_forward=tk.Button(time_dir, text=">", width=10, font=('MSゴシック', '15', "bold"), command=self.forward)
        self.b_forward.pack(side = tk.LEFT, padx=2, pady=2)
        self.b_forward.config(state=tk.DISABLED)
        self.key_forward_is_enable = False
        self.master.bind("<Right>", lambda event:self.key_forward())

        # カードテキスト
        card_text_dir = tk.LabelFrame(left_dir, text='Card Text', height=Const.CARD_TEXT_DIR_HEIGHT)
        card_text_dir.propagate(False)
        card_text_dir.pack(anchor=tk.W, padx=2, pady=2, fill=tk.X)
        self.card_text_manager = CardTextManager(self, card_text_dir)

        ################################################################################################################
        # 右（カードリストも早めに定義しておかなくちゃいけない）
        # チェーンスタック
        chain_dir = tk.LabelFrame(right_dir, text='Chain Stack', height = Const.CHAIN_DIR_HEIGHT)
        chain_dir.propagate(False)
        chain_dir.pack(anchor=tk.W, padx=2, pady=2, fill=tk.X)
        self.chain_manager = ChainManager(self, chain_dir)

        # カードリスト
        card_list_dir = tk.LabelFrame(right_dir, text='Card List')
        card_list_dir.pack(anchor=tk.W, padx=2, pady=2, expand=True, fill=tk.BOTH)
        self.card_list_manager = CardListManager(self, card_list_dir)


        ################################################################################################################
        # 中央
        # p1の手札
        rival_hand_dir = tk.LabelFrame(mid_dir, text='Rival Hand', height=Const.HAND_DIR_HEIGHT)
        rival_hand_dir.propagate(False)
        rival_hand_dir.pack(anchor=tk.W, padx=2, pady=2, fill=tk.X)
        self.rival_hand_manager = HandManager(self,c.enums.PlayerId.RIVAL, rival_hand_dir)

        # 盤面
        board_dir = tk.LabelFrame(mid_dir, text='Board')
        board_dir.pack(anchor=tk.W, padx=2, pady=2, expand=True, fill=tk.BOTH)
        self.board_manager = BoardManager(self, board_dir)

        # p0の手札
        my_hand_dir = tk.LabelFrame(mid_dir, text='My Hand', height=Const.HAND_DIR_HEIGHT)
        my_hand_dir.propagate(False)
        my_hand_dir.pack(anchor=tk.W, padx=2, pady=2, fill=tk.X)
        self.my_hand_manager = HandManager(self,c.enums.PlayerId.MYSELF, my_hand_dir)

        ################################################################################################################
        # 追加
        # コンテキスト
        context_dir = tk.LabelFrame(additional_dir, text='Command Log', height=Const.CONTEXT_DIR_HEIGHT)
        context_dir.propagate(False)
        context_dir.pack(anchor=tk.W, padx=2, pady=2, fill=tk.X)
        self.context_manager = ContextManager(self, context_dir)

        # デュエルログ
        log_dir = tk.LabelFrame(additional_dir, text='Duel Log Data')
        log_dir.pack(anchor=tk.W, padx=2, pady=2, expand=True, fill=tk.BOTH)
        self.log_manager = LogManager(self, log_dir)

        ################################################################################################################

        self.is_ready = True

    def back(self):
        self.is_ready = False
        if self.time > 0:
            self.time -= 1
        self.gui_update()
        # 巻き戻しのときは常にコマンドボタンを実行不可にする
        self.command_manager.disable_command_label()
        
    def forward(self):
        if self.time < len(self.memories) -1 :
            self.time += 1
        self.gui_update()
        # 次送りのときは最新以外ではコマンドボタンを実行不可にする
        if self.time < len(self.memories) -1 :
            self.command_manager.disable_command_label()
        else:
            # 最新時刻のときはis_ready = Trueにしてデータ受け取りを有効化する
            self.is_ready = True

    def key_back(self):
        if self.key_back_is_enable:
            self.back()

    def key_forward(self):
        if self.key_forward_is_enable:
            self.forward()

    # TODO:一時停止・再開を実装したが、若干動作が不安定
    def pause(self):
        self.is_ready = False

    def resume(self):
        self.time = len(self.memories) -1
        self.gui_update()
        self.is_ready = True

    def set_queue(self, command_num):
        if self.command_queue is not None:
            if self.command_queue.empty():
                self.command_queue.put(command_num)

                # 巻き戻したときに選んだ行動を強調表示するために保存
                self.memories[-1]["selected_command"] = command_num

                # 実行成功時はTrueを返す
                return True
            
        # 実行失敗時はFalseを返す
        return False


    def update(self, udi_log_data : mdl.UdiLogData):
        # memory更新
        memory = {
            "duel_log_data"    : copy.deepcopy(udi_log_data.duel_log_data),
            "command_request"  : copy.deepcopy(udi_log_data.command_request),
            "duel_state_data"  : copy.deepcopy(udi_log_data.duel_state_data),
            "selected_command" : -1, # コマンド送信があるときに別途設定するため、ここでは-1
            "ai_command"       : None, # set_ai_infoで別途設定するため、ここではNone
            "ai_info"          : None, # 同上
            "before_log_length": None, # このあと一個前のmemoryのlogを見て長さを取得する
            "before_command_log_length" : None, # コマンド送信があるときに別途設定するため、ここではNone
        }

        # 既存データを読み込むときなど，選択したコマンドの情報があるときは設定する
        if udi_log_data.selected_command >= 0:
            memory["selected_command"] = udi_log_data.selected_command

        if self.time >= 0:
            # このタイミングではself.timeはmemoryの最新(=直前のmemory)を指している
            memory["before_log_length"] = len(self.memories[self.time]["duel_log_data"])
            before_command_request : mdl.CommandRequest = self.memories[self.time]["command_request"]
            if len(before_command_request.commands) > 0:
                # コマンド要求があるときは，logの長さを取得
                memory["before_command_log_length"] = len(self.memories[self.time]["duel_log_data"])
            else:
                # 要求がないときは，前回のmemoryのコマンドログ長をそのまま引き継ぐ
                memory["before_command_log_length"] = self.memories[self.time]["before_command_log_length"]

        self.memories.append(memory)
        self.time += 1

        # gui更新
        self.gui_update()

        # ここで，デュエルが終了したタイミングだったら，そのデュエルのmemoryを保存して次のmemoryを初期化
        if len(udi_log_data.duel_log_data) > 0:
            latest_log = udi_log_data.duel_log_data[-1]
            if latest_log.type == c.DuelLogType.DUEL_END:
                # duel_end時は，infoを追加し，memorisを保存して次のmemoryを初期化
                ld = latest_log.data
                duel_end_log = mdl.DuelEndData(ld)
                info = f"Duel{len(self.memories_info_list)}:result={duel_end_log.result_type}, fin_type={duel_end_log.finish_type}, "
                self.memories_info_list.append(info)

                self.memories_list.append(self.memories)
                self.memories = []
                self.time = -1

                # gui_managerのresetを呼ぶ(gui_managerは内部でlogを保持しているので，次のデュエルに移るときにresetしておかないと前のデュエルのデータが残る)
                self.log_manager.reset()


    def set_ai_info(self, ai_command, ai_info):
        # 最新のmemoryのai_command, ai_infoを置き換える
        self.memories[-1]["ai_command"] = copy.deepcopy(ai_command)
        self.memories[-1]["ai_info"]    = copy.deepcopy(ai_info)
        # gui更新
        self.gui_update()


    def gui_update(self):
        memory = self.memories[self.time]
        duel_log_data                        = memory["duel_log_data"   ]
        command_request : mdl.CommandRequest = memory["command_request" ]
        duel_state_data                      = memory["duel_state_data" ]
        selected_command                     = memory["selected_command"]
        ai_command                           = memory["ai_command"      ]
        ai_info                              = memory["ai_info"         ]
        before_log_length                    = memory["before_log_length"]
        before_command_log_length            = memory["before_command_log_length"]
        
        # gui更新
        self.normal_update(duel_log_data, duel_state_data)

        # ログの差分を強調表示
        if before_log_length is not None:
            self.log_manager.highlight_log_diff(before_log_length, before_command_log_length)
        
        # commandが絡むときの更新
        if len(command_request.commands) > 0:
            self.command_update(duel_state_data, command_request)
            # 過去選択していたコマンドをハイライト
            if selected_command >= 0:
                self.command_manager.highlight_selected_command(selected_command)
            # プレイAIによるコマンドをハイライト
            if ai_command is not None:
                self.command_manager.highlight_ai_command(ai_command)
            # プレイAIによる情報を表示
            if ai_info is not None:
                self.command_manager.display_ai_info(ai_info)
            

        # 巻き戻しボタンなどの実行不可更新
        if self.time > 0:
            self.b_back.config(state=tk.NORMAL)
            self.key_back_is_enable = True
        else:
            self.b_back.config(state=tk.DISABLED)
            self.key_back_is_enable = False
        if self.time < len(self.memories) -1 :
            self.b_forward.config(state=tk.NORMAL)
            self.key_forward_is_enable = True
        else:
            self.b_forward.config(state=tk.DISABLED)
            self.key_forward_is_enable = False

        if self.is_ready:
            self.b_pause.config(state=tk.NORMAL)
            self.b_resume.config(state=tk.DISABLED)
        else:
            self.b_pause.config(state=tk.DISABLED)
            self.b_resume.config(state=tk.NORMAL)
        
        # queueが無いときは、GUIでコマンド実行をさせない
        if self.command_queue is None:
            self.command_manager.disable_command_label()


    # commandsが絡まない部分のgui更新
    def normal_update(self, duel_log_data, duel_state_data):
        self.card_list_manager.set_duel_card_table(duel_state_data)
        self.card_text_manager.set_duel_card_table(duel_state_data)

        self.chain_manager.update(duel_state_data)
        self.rival_hand_manager.update(duel_state_data)
        self.board_manager.update(duel_state_data)
        self.my_hand_manager.update(duel_state_data)
        self.log_manager.update(duel_log_data)

        self.command_manager.reset()
        self.context_manager.reset()


    # command_requestが絡む部分のgui更新
    def command_update(self, duel_state_data, command_request):
        self.command_manager.update(command_request, duel_state_data)
        self.context_manager.update(command_request, duel_state_data)
        self.dialog_manager.update(command_request)


    # ファイルを開く
    def open_file(self):
        # ファイル選択ダイアログの表示
        file_path = tk.filedialog.askopenfilename(
            filetypes=[
                ("udi_log_data", "*.gz"),
            ],
            initialdir="./"
        )

        if len(file_path) != 0:
            # 読み込み中を示すウィンドウを作成
            loading_window = tk.Toplevel(self.master)
            loading_window.title("Loading")
            loading_window.geometry("200x100")
            loading_label = tk.Label(loading_window, text="Loading...")
            loading_label.pack()
            loading_window.grab_set()
            loading_window.focus_set()
            loading_window.update()

            # ファイルを読み込む
            self._load_file(file_path)

            # さっきのウィンドウで読み込み終了を示す
            fin_button = tk.Button(loading_window, text="Finish", command=loading_window.destroy)
            fin_button.pack()

        else:
            # ファイル選択がキャンセルされた場合は何もしない
            pass


    # フォルダを開く
    def open_folder(self):
        # フォルダ選択ダイアログの表示
        folder_path = tk.filedialog.askdirectory(
            initialdir="./"
        )

        if len(folder_path) != 0:
            # 読み込み中を示すウィンドウを作成
            loading_window = tk.Toplevel(self.master)
            loading_window.title("Loading")
            loading_window.geometry("200x100")
            loading_label = tk.Label(loading_window, text="Loading...")
            loading_label.pack()
            loading_window.grab_set()
            loading_window.focus_set()
            loading_window.update()

            # 指定したフォルダ内の全ての.json.gzファイルを読み込む
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".json.gz"):
                    file_path = os.path.join(folder_path, file_name)
                    self._load_file(file_path)

            # さっきのウィンドウで読み込み終了を示す
            fin_button = tk.Button(loading_window, text="Finish", command=loading_window.destroy)
            fin_button.pack()

        else:
            # フォルダ選択がキャンセルされた場合は何もしない
            pass


    # file_pathを指定して，ファイルを読み込む
    def _load_file(self, file_path):
        # ファイルが選択された場合
        udi_log_data_list = UdiLogUtil.load_udi_log(file_path)

        # ファイル名を抽出
        file_name = file_path.split("/")[-1]

        # 先攻後攻の情報を抽出
        first_duel_log = udi_log_data_list[0].duel_log_data[0]
        if first_duel_log.type == c.DuelLogType.CARD_MOVE:
            first_card_move_log = mdl.CardMoveData(first_duel_log.data)
            first_player = first_card_move_log.from_player_id
        else:
            first_player = c.PlayerId.NO_VALUE

        # デュエル単位でmemoryを保存
        memories = []
        duel_count = 0
        before_log_length = -1
        before_command_log_length = -1
        for udi_log_data in udi_log_data_list:
            memory = {
                "duel_log_data"    : udi_log_data.duel_log_data,
                "command_request"  : udi_log_data.command_request,
                "duel_state_data"  : udi_log_data.duel_state_data,
                "selected_command" : udi_log_data.selected_command,
                "ai_command"       : None,
                "ai_info"          : None,
                "before_log_length": None,
                "before_command_log_length" : None
            }
            if before_log_length >= 0:
                memory["before_log_length"] = before_log_length
            if before_command_log_length >= 0:
                memory["before_command_log_length"] = before_command_log_length
            memories.append(memory)

            # logの長さを保存しておく
            before_log_length = len(udi_log_data.duel_log_data)
            if len(udi_log_data.command_request.commands) > 0:
                before_command_log_length = len(udi_log_data.duel_log_data)
                
            latest_log = udi_log_data.duel_log_data[-1]
            if latest_log.type == c.DuelLogType.DUEL_END:
                # duel_end時は，infoを追加し，memorisを保存して次のmemoryを初期化
                ld = latest_log.data
                duel_end_log = mdl.DuelEndData(ld)
                info = f"{file_name} : Duel{duel_count} : first_player{first_player}, result={duel_end_log.result_type}, fin_type={duel_end_log.finish_type}, "
                self.memories_info_list.append(info)
                duel_count += 1

                self.memories_list.append(memories)
                memories = []
                before_log_length = -1
                before_command_log_length = -1

        # デュエルエンドまで行ってないかもしれないので，最後のmemoryを保存
        if len(memories) > 0:
            info = f"{file_name} : Duel{duel_count} : first_player{first_player}, result={c.ResultType.NONE}, fin_type={c.ResultType.NONE}, "
            self.memories_info_list.append(info)
            self.memories_list.append(memories)


    # デュエルを指定して再生
    def load_duel(self):
        # 新規ウィンドウを表示して，現在のmemories_listを表示する
        duel_window = tk.Toplevel(self.master)
        duel_window.title("Select Duel")
        duel_window.geometry("500x300")
        duel_window.grab_set()
        duel_window.focus_set()

        list_dir = tk.Frame(duel_window)
        list_dir.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # リストボックスを作成
        duel_listbox = tk.Listbox(list_dir, selectmode=tk.SINGLE, width = 50)
        for i, info in enumerate(self.memories_info_list):
            duel_listbox.insert(tk.END, info)
        # スクロールバーを作成
        duel_scroll = tk.Scrollbar(list_dir, orient=tk.VERTICAL, command=duel_listbox.yview)
        duel_listbox['yscrollcommand'] = duel_scroll.set

        # リストボックスの真横にスクロールバーを配置
        duel_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        duel_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 選択ボタンを作成
        select_button = tk.Button(duel_window, text="Select", command=lambda: self.select_duel(duel_listbox.curselection(), duel_window))
        select_button.pack(side=tk.BOTTOM, ipadx = 30, ipady = 10, pady = 10)


    def select_duel(self, duel_nums, duel_window):
        # curselection()はタプルで返ってくる．選択がない場合は空のタプルが返る．
        if len(duel_nums) == 0:
            return
        duel_num = duel_nums[0]

        # 選択されたデュエルのmemoryを取得
        self.memories = self.memories_list[duel_num]
        self.time = 0
        # gui_updateのまえに，log_managerをresetしておく(gui_managerは内部でlogを保持しているので，次のデュエルに移るときにresetしておかないと前のデュエルのデータが残る)
        self.log_manager.reset()
        self.gui_update()

        # 選択ウィンドウを閉じる
        duel_window.destroy()


    # ファイル読み込みについて
    def about_file(self):
        # 新規ウィンドウを表示して，ファイル読み込みについての説明を表示する
        file_window = tk.Toplevel(self.master)
        file_window.title("About File")
        file_window.geometry("500x300")
        file_window.grab_set()
        file_window.focus_set()

        # 説明文を表示
        text  = "ファイル→ファイルを読み込む：あらかじめudi_io.flush_udi_logsで保存したログ（.json.gz）を読み込みます．なお，この段階ではデュエルはまだ表示されません．\n"
        text  = "ファイル→フォルダを読み込む：指定したフォルダ内の.json.gzファイルを全て読み込みます．\n"
        text += "ファイル→デュエルを指定して再生：読み込んだログの中からデュエルを選択して再生します．\n"
        text += "\n"
        text += "なお，本GUIはファイル読み込みではなくリアルタイムにデュエルを見ることもできます．(udi_gui_thread.pyを参照)\n"
        text += "その際，GUI内部でデュエルごとのログを保持しているので，同様に過去のデュエルを選択・再生できます．\n"
        text += "ただし，GUIはログの保存に対応していないので，別途udi_io.flush_udi_logsで保存してください．\n"
        text += "！！注意！！：過去のデュエルを再生した時点で，現在のデュエルには復帰できません．\n"
        label = tk.Label(file_window, text=text)
        label.pack()


    # 画面の見方について
    def about_display(self):
        pass

    # コマンド実行について
    def about_exec_command(self):
        # 新規ウィンドウを表示して，コマンド実行についての説明を表示する
        command_window = tk.Toplevel(self.master)
        command_window.title("About Command")
        command_window.geometry("500x300")
        command_window.grab_set()
        command_window.focus_set()

        # 説明文を表示
        text  = "本GUIは，起動時にqueueを渡すことで，udi_ioにコマンドを渡すことができます．（udi_gui_thread.pyを参照）\n"
        text += "ただし，かなり簡易的な実装なため，不具合が発生する可能性もあります．特に，過去のデュエルを見るときに発生しうるので注意してください．\n"
        text += "！！注意！！：過去のデュエルを再生した時点で，現在のデュエルには復帰できません．\n"
        label = tk.Label(command_window, text=text)
        label.pack()



if __name__ == '__main__':
    f = UdiGUIFrame()
    f.pack()
    f.mainloop()