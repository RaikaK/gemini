import pickle
import enum
import collections
import dataclasses
from pathlib import Path
from typing import Dict, Set, List, Any, Tuple

import src.config as config
from src.agents.noah.handlers.utils.card_map import CARD_MAP

# 逆引き辞書 (ID -> カード名)
CARD_MAP_REVERSE = {info["id"]: name for name, info in CARD_MAP.items()}

# 調査対象の全CommandEntry変数
ALL_COMMAND_FIELDS = [
    "command_type",
    "player_id",
    "pos_id",
    "card_index",
    "card_id",
    "effect_card_id",
    "effect_no",
    "phase",
    "dialog_text_id",
    "stand_face",
    "stand_turn",
    "coin_face",
    "card_attribute",
    "species",
    "number",
    "yes_no",
    "table_index",
]

# DecisionMaker のハンドラ名対応（表示用）
FILE_NAME_MAP = {
    (0, -1): "draw_phase",
    (1, -1): "main_phase",
    (2, -1): "battle_phase",
    (4, -1): "attack_target",
    (5, -1): "activate_confirmation",
    (6, 0): "summon_release",
    (9, 7): "chain_before_send_to_grave",
    (9, 12): "chain_before_target",
    (9, 14): "chain_before_monster",
    (9, 15): "chain_before_destroy",
    (9, 23): "chain_before_discard",
    (9, 27): "chain_before_face_down",
    (10, 1): "chain_effect_set_query",
    (10, 2): "chain_effect_set_card",
    (10, 3): "chain_effect_special_summon_deck",
    (10, 4): "chain_effect_search_deck",
    (10, 6): "chain_effect_send_to_grave",
    (10, 18): "chain_effect_ritual_select",
    (10, 19): "chain_effect_ritual_release",
    (10, 24): "chain_effect_position",
    (10, 29): "chain_effect_special_summon",
    (10, 30): "chain_effect_special_summon_hand",
    (12, 20): "other_rewind_attack",
}


def get_val_raw(cmd, field_name: str) -> Any:
    """コマンドオブジェクトから生の値を取得するヘルパー"""
    try:
        val = getattr(cmd, field_name)
        return val.value if isinstance(val, enum.Enum) else val
    except AttributeError:
        return None


def main():
    # 1. 全データを読み込み、(Type, ID) ごとに全てのコマンドを1つのリストに集約（和集合を作成）
    print("Loading all demonstration data...")

    # 和集合用の辞書: key -> 全ての履歴に登場したコマンドのフラットなリスト
    grouped_all_commands = collections.defaultdict(list)

    # configのパスを確認
    target_dirs = [config.DEMONSTRATION_DIR]
    if hasattr(config, "OLD_DEMONSTRATION_DIR"):
        target_dirs.append(config.OLD_DEMONSTRATION_DIR)

    pkl_files = []
    for d in target_dirs:
        if d.exists():
            pkl_files.extend(list(d.glob("*.pkl")))

    if not pkl_files:
        print("No pickle files found.")
        return

    count_files = 0
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, "rb") as f:
                demos = pickle.load(f)
                count_files += 1
                for d in demos:
                    req = d["state"].command_request
                    # キー: (selection_type, selection_id)
                    key = (int(req.selection_type), int(req.selection_id))

                    # CommandRequestに含まれる「全ての選択肢」を、そのType/IDの巨大リストに追加
                    grouped_all_commands[key].extend(req.commands)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")

    # 2. グループごとに「全データの和集合」における差分チェックを実行
    print(f"\nAnalyzed {count_files} files.")
    print(f"Found {len(grouped_all_commands)} distinct decision types.")
    print("Checking variables across the UNION of all historical options...\n")
    print("=" * 80)

    # キーをソートして順番に出力
    for (stype, sid), all_cmds in sorted(grouped_all_commands.items()):

        # 全フィールドの値を収集するためのセット
        field_values_map = {field: set() for field in ALL_COMMAND_FIELDS}

        # 蓄積した全てのコマンド（和集合）を走査
        for cmd in all_cmds:
            for field in ALL_COMMAND_FIELDS:
                val = get_val_raw(cmd, field)
                field_values_map[field].add(val)

        # 値が2種類以上ある（＝全履歴の中で変化があった）変数を特定
        varying_fields = [f for f in ALL_COMMAND_FIELDS if len(field_values_map[f]) > 1]

        # ハンドラ名取得
        handler_name = FILE_NAME_MAP.get((stype, sid), "Unknown Handler")

        # ターミナル出力
        print(f"Type: {stype:<2} | ID: {sid:<3} | {handler_name}")
        if varying_fields:
            # カンマ区切りで見やすく表示
            print(f"  -> Varying Vars: {', '.join(varying_fields)}")
        else:
            print(f"  -> Varying Vars: (None - Strictly constant across all history)")
        print("-" * 80)


if __name__ == "__main__":
    main()
