import tkinter as tk

from ygo.gui.manager.scollable_frame import ScrollableFrameY, ScrollableFrameX

BASE_SCROLL_REGION = 1800


class ScrollerY(ScrollableFrameY):
    """
    スクローラY
    """

    def __init__(self, master: tk.Misc, udi_gui_frame, **key) -> None:
        """
        初期化する。
        """
        self.udi_gui_frame = udi_gui_frame

        ScrollableFrameY.__init__(self, master=master, **key)

    def make_canvas_frame(self) -> None:
        """
        キャンバスを作成する。
        """
        factor: float = self.udi_gui_frame.factor
        scaled_region: int = int(BASE_SCROLL_REGION * factor)

        self.canvas: tk.Canvas = tk.Canvas(self.master, **self.key)
        self.frame: tk.Frame = tk.Frame(self.canvas, **self.key)

        # Canvasを親とした縦方向のScrollbar
        self.scrollbar: tk.Scrollbar = tk.Scrollbar(self.canvas, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(scrollregion=(0, 0, scaled_region, scaled_region))
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.frame.pack(expand=True, fill=tk.BOTH)

        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.frame.bind("<MouseWheel>", self._on_mousewheel)

        self.frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))


class ScrollerX(ScrollableFrameX):
    """
    スクローラX
    """

    def __init__(self, master: tk.Misc, udi_gui_frame, **key) -> None:
        """
        初期化する。
        """
        self.udi_gui_frame = udi_gui_frame

        super().__init__(master, **key)

    def make_canvas_frame(self) -> None:
        """
        キャンバスを作成する。
        """
        factor: float = self.udi_gui_frame.factor
        scaled_region: int = int(BASE_SCROLL_REGION * factor)

        self.canvas: tk.Canvas = tk.Canvas(self.master, **self.key)
        self.frame: tk.Frame = tk.Frame(self.canvas, **self.key)

        # Canvasを親とした縦方向のScrollbar
        self.scrollbar: tk.Scrollbar = tk.Scrollbar(self.canvas, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(scrollregion=(0, 0, scaled_region, scaled_region))
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.frame.pack(expand=True, fill=tk.BOTH)

        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.frame.bind("<MouseWheel>", self._on_mousewheel)

        self.frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
