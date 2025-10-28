import PIL.Image as Img
import PIL.ImageTk as Itk
import tkinter as tk

from ygo import constants as c
from ygo.gui.manager.board_manager import BoardManager, CardLabel, PositionLabel
from ygo.gui.manager.const import Const


class GUIPositionLabel(PositionLabel):
    """
    GUIポジションラベル
    """

    def __init__(
        self,
        master: tk.Misc,
        img: Itk.PhotoImage,
        player_id: int,
        pos_id: int,
        udi_gui_frame,
        factor: float,
    ) -> None:
        """
        初期化する。
        """
        super().__init__(master, img, player_id, pos_id, udi_gui_frame)

        scaled_font: tuple = (
            Const.BOARD_POSITION_FONT[0],
            max(8, int(int(Const.BOARD_POSITION_FONT[1]) * factor)),
            Const.BOARD_POSITION_FONT[2],
        )
        scaled_pady: int = max(1, int(2 * factor))

        self.text_label.config(font=scaled_font)
        self.text_label.pack_configure(pady=scaled_pady)


class GUIBoard(BoardManager):
    """
    GUIボード
    """

    def __init__(self, udi_gui_frame, master: tk.Misc, **key) -> None:
        """
        初期化する。
        """
        factor: float = udi_gui_frame.factor

        self.udi_gui_frame = udi_gui_frame
        self.master: tk.Misc = master
        self.key: dict = key

        # 各マスの背景画像
        card_width: int = int(Const.S_CARD_W * factor)
        card_height: int = int(Const.S_CARD_H * factor)
        bg_img: Img.Image = Img.new("RGB", (card_width, card_height), "gray")
        bg_tkimg: Itk.PhotoImage = Itk.PhotoImage(bg_img)

        # カード関連
        self.board: dict = {c.enums.PlayerId.MYSELF: dict(), c.enums.PlayerId.RIVAL: dict()}
        row_col: dict = {
            c.enums.PlayerId.MYSELF: {
                c.enums.PosId.MONSTER_L_L: (5, 2),
                c.enums.PosId.MONSTER_L: (5, 3),
                c.enums.PosId.MONSTER_C: (5, 4),
                c.enums.PosId.MONSTER_R: (5, 5),
                c.enums.PosId.MONSTER_R_R: (5, 6),
                c.enums.PosId.EX_L_MONSTER: (4, 3),
                c.enums.PosId.EX_R_MONSTER: (4, 5),
                c.enums.PosId.MAGIC_L_L: (6, 2),
                c.enums.PosId.MAGIC_L: (6, 3),
                c.enums.PosId.MAGIC_C: (6, 4),
                c.enums.PosId.MAGIC_R: (6, 5),
                c.enums.PosId.MAGIC_R_R: (6, 6),
                c.enums.PosId.FIELD: (5, 1),
            },
            c.enums.PlayerId.RIVAL: {
                c.enums.PosId.MONSTER_L_L: (3, 6),
                c.enums.PosId.MONSTER_L: (3, 5),
                c.enums.PosId.MONSTER_C: (3, 4),
                c.enums.PosId.MONSTER_R: (3, 3),
                c.enums.PosId.MONSTER_R_R: (3, 2),
                c.enums.PosId.EX_L_MONSTER: (4, 5),
                c.enums.PosId.EX_R_MONSTER: (4, 3),
                c.enums.PosId.MAGIC_L_L: (2, 6),
                c.enums.PosId.MAGIC_L: (2, 5),
                c.enums.PosId.MAGIC_C: (2, 4),
                c.enums.PosId.MAGIC_R: (2, 3),
                c.enums.PosId.MAGIC_R_R: (2, 2),
                c.enums.PosId.FIELD: (3, 7),
            },
        }
        for player_id in range(c.enums.PlayerId.UPPER_VALUE):
            for pos_id in range(c.enums.PosId.FIELD + 1):
                label: CardLabel = CardLabel(self.master, bg_tkimg, self.udi_gui_frame)
                label_pos: tuple = row_col[player_id][pos_id]

                if pos_id in [c.enums.PosId.EX_L_MONSTER, c.enums.PosId.EX_R_MONSTER]:
                    if player_id == c.enums.PlayerId.MYSELF:
                        label.grid(row=label_pos[0], column=label_pos[1])
                        self.board[c.enums.PlayerId.MYSELF][pos_id] = label
                        if pos_id == c.enums.PosId.EX_L_MONSTER:
                            self.board[c.enums.PlayerId.RIVAL][c.enums.PosId.EX_R_MONSTER] = label
                        elif pos_id == c.enums.PosId.EX_R_MONSTER:
                            self.board[c.enums.PlayerId.RIVAL][c.enums.PosId.EX_L_MONSTER] = label
                    else:
                        pass
                else:
                    label.grid(row=label_pos[0], column=label_pos[1])
                    self.board[player_id][pos_id] = label

        # カードリスト関連
        self.card_list_position: dict = {c.enums.PlayerId.MYSELF: dict(), c.enums.PlayerId.RIVAL: dict()}
        row_col: dict = {
            c.enums.PlayerId.MYSELF: {
                c.enums.PosId.EXCLUDE: (8, 4),
                c.enums.PosId.GRAVE: (8, 5),
                c.enums.PosId.DECK: (8, 6),
                c.enums.PosId.EXTRA: (8, 7),
            },
            c.enums.PlayerId.RIVAL: {
                c.enums.PosId.EXCLUDE: (0, 4),
                c.enums.PosId.GRAVE: (0, 5),
                c.enums.PosId.DECK: (0, 6),
                c.enums.PosId.EXTRA: (0, 7),
            },
        }
        for player_id in range(c.enums.PlayerId.UPPER_VALUE):
            for pos_id in range(c.enums.PosId.EXTRA, c.enums.PosId.EXCLUDE + 1):
                label: GUIPositionLabel = GUIPositionLabel(
                    self.master, bg_tkimg, player_id, pos_id, self.udi_gui_frame, factor
                )
                label_pos = row_col[player_id][pos_id]
                label.grid(row=label_pos[0], column=label_pos[1])
                self.card_list_position[player_id][pos_id] = label

        scaled_pady: int = max(1, int(4 * factor))
        scaled_player_font: tuple = (
            Const.BOARD_PLAYER_TEXT_FONT[0],
            max(8, int(int(Const.BOARD_PLAYER_TEXT_FONT[1]) * factor)),
            Const.BOARD_PLAYER_TEXT_FONT[2],
        )
        scaled_lp_font: tuple = (
            Const.BOARD_PLAYER_LP_FONT[0],
            max(8, int(int(Const.BOARD_PLAYER_LP_FONT[1]) * factor)),
            Const.BOARD_PLAYER_LP_FONT[2],
        )
        scaled_info_font: tuple = (
            Const.BOARD_PLAYER_INFO_FONT[0],
            max(8, int(int(Const.BOARD_PLAYER_INFO_FONT[1]) * factor)),
            Const.BOARD_PLAYER_INFO_FONT[2],
        )
        scaled_phase_font: tuple = (
            Const.BOARD_PHASE_FONT[0],
            max(8, int(int(Const.BOARD_PHASE_FONT[1]) * factor)),
            Const.BOARD_PHASE_FONT[2],
        )

        # RIVAL関連
        self.rival_text_label: tk.Label = tk.Label(
            self.master, text="相手", font=scaled_player_font, relief="ridge", bg="lightgray"
        )
        self.rival_text_label.grid(row=0, column=1, pady=scaled_pady)

        self.rival_lp_label: tk.Label = tk.Label(
            self.master, text="LP\n8000", font=scaled_lp_font, relief="ridge", bg="lightgray"
        )
        self.rival_lp_label.grid(row=0, column=2, pady=scaled_pady)

        self.rival_info_label: tk.Label = tk.Label(
            self.master, text="召喚権:1", font=scaled_info_font, relief="ridge", bg="lightgray"
        )
        self.rival_info_label.grid(row=0, column=3, pady=scaled_pady)

        # MYSELF関連
        self.myself_text_label: tk.Label = tk.Label(
            self.master, text="自分", font=scaled_player_font, relief="ridge", bg="lightgray"
        )
        self.myself_text_label.grid(row=8, column=1, pady=scaled_pady)

        self.myself_lp_label: tk.Label = tk.Label(
            self.master, text="LP\n8000", font=scaled_lp_font, relief="ridge", bg="lightgray"
        )
        self.myself_lp_label.grid(row=8, column=2, pady=scaled_pady)

        self.myself_info_label: tk.Label = tk.Label(
            self.master, text="召喚権:1", font=scaled_info_font, relief="ridge", bg="lightgray"
        )
        self.myself_info_label.grid(row=8, column=3, pady=scaled_pady)

        # 共通関連
        self.phase_label: tk.Label = tk.Label(
            self.master, text="フェーズ:\n \n ", font=scaled_phase_font, relief="ridge", bg="lightgray"
        )
        self.phase_label.grid(row=9, column=3, columnspan=3, pady=scaled_pady)
