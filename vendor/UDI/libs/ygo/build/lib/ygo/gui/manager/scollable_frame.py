#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Konami Digital Entertainment Co., Ltd. All rights reserved.

import tkinter as tk


class ScrollableFrameY:
    def __init__(self, master, **key):
        self.master=master
        self.key = key

        self.make_canvas_frame()
        self.once = False


    def make_canvas_frame(self):
        self.canvas = tk.Canvas(self.master, **self.key)
        self.frame=tk.Frame(self.canvas, **self.key)

        # Canvasを親とした縦方向のScrollbar
        self.scrollbar = tk.Scrollbar(
            self.canvas, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.canvas.configure(scrollregion=(0, 0, 1800, 1800))
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.frame.pack(expand=True, fill=tk.BOTH)

        # Canvas上の座標(0, 0)に対してFrameの左上（nw=north-west）をあてがうように、Frameを埋め込む
        # self.canvas.create_window((0, 0), window=self.frame, anchor="nw", width=1800, height=1800)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.frame.bind("<MouseWheel>", self._on_mousewheel)

        # スクロール範囲自動調整
        self.frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")


class ScrollableFrameX:
    def __init__(self, master, **key):
        self.master=master
        self.key = key

        self.make_canvas_frame()
        self.once = False


    def make_canvas_frame(self):
        self.canvas = tk.Canvas(self.master, **self.key)
        self.frame=tk.Frame(self.canvas, **self.key)

        # Canvasを親とした縦方向のScrollbar
        self.scrollbar = tk.Scrollbar(
            self.canvas, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        self.canvas.configure(scrollregion=(0, 0, 1800, 1800))
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.frame.pack(expand=True, fill=tk.BOTH)

        # Canvas上の座標(0, 0)に対してFrameの左上（nw=north-west）をあてがうように、Frameを埋め込む
        # self.canvas.create_window((0, 0), window=self.frame, anchor="nw", width=1800, height=1800)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.frame.bind("<MouseWheel>", self._on_mousewheel)

        # スクロール範囲自動調整
        self.frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

    def _on_mousewheel(self, event):
        self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
