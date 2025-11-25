from queue import Queue

from ygo.gui.udi_gui_thread import UdiGUIThread

from src.gui.gui_frame import GUIFrame


class GUIThread(UdiGUIThread):
    """
    GUIスレッド
    """

    def __init__(self) -> None:
        """
        初期化する。
        """
        super().__init__()
        self.udi_gui_frame: GUIFrame | None = None

    def _start_thread(self, queue: Queue) -> None:
        """
        スレッドを開始する。
        """
        self.udi_gui_frame = GUIFrame(queue=queue)
        self.udi_gui_frame.pack()
        self.udi_gui_frame.mainloop()
