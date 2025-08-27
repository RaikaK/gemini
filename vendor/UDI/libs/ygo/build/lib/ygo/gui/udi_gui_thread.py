#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import threading
import time

from .udi_gui_frame import UdiGUIFrame
from ygo import models as mdl

class UdiGUIThread:
    def start(self, queue=None):
        t = threading.Thread(target=self._start_thread, daemon=True, args=(queue,))
        t.start()

    def _start_thread(self, queue):
        self.udi_gui_frame = UdiGUIFrame(queue=queue)
        self.udi_gui_frame.pack()
        self.udi_gui_frame.mainloop()
        

    def set_data(self, duel_log_data, command_request, duel_state_data):
        # TODO: ここで受け取る時点で，udi_log_dataにしたほうがいい？
        while(True):
            if self.is_ready():
                udi_log_data = mdl.UdiLogData(command_request, duel_state_data, duel_log_data, -1)
                self.udi_gui_frame.update(udi_log_data)
                return True
            else:
                time.sleep(0.5)
    

    def set_ai_info(self, ai_command:int, ai_info:list[str]):
        """プレイAIによる情報のみをセットする。

        Args:
            ai_command (int): プレイAIが選択した行動
            ai_info (list[str]): プレイAIによるコマンドごとの情報(選択確率など)のリスト
        """
        while(True):
            if self.is_ready():
                self.udi_gui_frame.set_ai_info(ai_command, ai_info)
                return True
            else:
                time.sleep(0.5)


    def is_ready(self):
        while(True):
            try:
                if self.udi_gui_frame is not None:
                    return self.udi_gui_frame.is_ready
                else: 
                    time.sleep(0.5)
            except:
                time.sleep(0.5)
                pass

if __name__ == "__main__":
    import traceback
    from queue import Queue
    from ygo.udi_io import UdiIO

    # GUIアプリとメインスレッドをつなぐコマンド用キュー
    q = Queue(1)

    gui_thread = UdiGUIThread()
    gui_thread.start(q)

    udi_io = UdiIO(api_version=1, tcpport=8573)
    while(True):
        try:
            if (not udi_io.input()):
                pass

            else:
                # デュエル開始・終了チェック
                if udi_io.is_duel_start():
                    print("duel start")
                if udi_io.is_duel_end():
                    print("duel end")
                    print(udi_io.get_duel_end_data())

                # 入力要求あり
                if udi_io.is_command_required():
                    # コマンド実行前にudi_guiにコマンド系データ反映
                    gui_thread.set_data(duel_log_data=udi_io.get_duel_log_data(), 
                                        command_request=udi_io.get_command_request(), 
                                        duel_state_data=udi_io.get_duel_state_data())
                    
                    # GUIアプリからのコマンド待ち
                    commands = udi_io.get_command_request().commands
                    while True:
                        action = q.get() # gui_threadのほうでキューにactionがsetされたら、getする
                        try:
                            action = int(action)
                        except ValueError:
                            print("Error : please enter int")
                        else:
                            if 0 <= action < len(commands):
                                break
                    
                    # コマンド決定したので、udi_ioで送信
                    udi_io.output_command(action)
                else:
                    # 入力要求なしでも、udi_guiにデータ反映
                    gui_thread.set_data(duel_log_data=udi_io.get_duel_log_data(),
                                        command_request=udi_io.get_command_request(),
                                        duel_state_data=udi_io.get_duel_state_data())

                
        except Exception as e:
            print('catch error:' + traceback.format_exc())
            print('exit python')
            exit()
        

