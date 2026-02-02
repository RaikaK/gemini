import argparse
import pickle
import enum
import os
import dataclasses
from pathlib import Path
from typing import Dict, Set, List, Any, Tuple

import src.config as config
from src.agents.noah.handlers.utils.card_map import CARD_MAP
from src.agents.noah.handlers.utils.situation import Situation
from ygo.constants.enums import PlayerId

# 逆引き辞書 (ID -> カード名)
CARD_MAP_REVERSE = {info["id"]: name for name, info in CARD_MAP.items()}

# 調査対象フィールド (基本セット)
CHECK_FIELDS = ["command_type", "player_id", "card_id", "effect_no", "phase", "yes_no", "stand_turn"]

# DecisionMaker のハンドラ名対応
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

# 解析対象のSituationプロパティ一覧
SITUATION_PROPS = [
    "my_hand_count",
    "my_deck_count",
    "my_grave_count",
    "my_mzone_count",
    "my_szone_count",
    "my_hand_monster_count",
    "rival_hand_count",
    "rival_deck_count",
    "rival_grave_count",
    "rival_mzone_count",
    "rival_szone_count",
    "my_monster_atk_max",
    "my_monster_def_max",
    "my_face_up_atk_monster_count",
    "my_face_up_def_monster_count",
    "my_face_down_monster_count",
    "my_face_down_spell_count",
    "my_activatable_spell_count",
    "rival_monster_atk_max",
    "rival_monster_def_max",
    "rival_monster_atk_min",
    "rival_monster_def_min",
    "rival_face_up_atk_monster_count",
    "rival_face_up_def_monster_count",
    "rival_face_down_monster_count",
    "rival_face_down_spell_count",
    "rival_activatable_spell_count",
    "is_my_monster_attacking",
    "is_my_monster_attacked",
    "is_rival_monster_attacking",
    "is_rival_monster_attacked",
    "my_attacking_monster_id",
    "my_attacking_monster_atk",
    "my_attacking_monster_def",
    "my_attacked_monster_id",
    "rival_attacking_monster_id",
    "rival_attacked_monster_id",
    "is_my_turn",
    "is_rival_turn",
    "turn_num",
    "is_first_turn",
    "my_lp",
    "rival_lp",
    "lp_diff",
    "lp_ratio",
    "is_lp_advantage",
    "is_lp_disadvantage",
    "is_lp_equal",
    "phase",
    "is_draw_phase",
    "is_standby_phase",
    "is_main1",
    "is_battle_phase",
    "is_main2",
    "is_end_phase",
    "is_phase_none",
    "is_main_phase",
    "step",
    "is_step_null",
    "is_step_start",
    "is_step_battle",
    "is_step_damage",
    "is_step_end",
    "dmg_step",
    "is_dmg_null",
    "is_dmg_start",
    "is_dmg_before_calc",
    "is_dmg_calc",
    "is_dmg_after_calc",
    "is_dmg_end",
    "my_summon_count",
    "rival_summon_count",
    "can_summon",
    "can_rival_summon",
    "chain_count",
    "is_chaining",
    "last_chain_card_id",
    "last_chain_player_id",
    "is_last_chain_my",
    "is_last_chain_rival",
    "is_last_chain_effect_1",
    "is_last_chain_effect_2",
    "is_last_chain_effect_3",
    "is_last_chain_before_activation",
    "is_last_chain_before_resolution",
    "is_last_chain_resolving",
    "first_chain_card_id",
    "first_chain_player_id",
    "is_first_chain_my",
    "is_first_chain_rival",
    "is_my_card_targeted",
    "is_rival_card_targeted",
    "is_my_card_targeted_anywhere",
    "is_rival_card_targeted_anywhere",
    "log_count",
    "last_action_card_id",
    "last_action_table_index",
    "last_action_selection_type",
    "last_action_selection_id",
    "last_action_card_face",
    "last_action_card_turn",
    "last_action_card_atk",
    "last_action_card_def",
    "last_action_card_used_effect1",
    "last_action_card_used_effect2",
    "last_action_card_equip_target",
]


@dataclasses.dataclass
class StrategyPattern:
    available_idents: Set[str]
    chosen_ident: str
    samples: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    is_merged: bool = False
    expanded_options: Set[str] = dataclasses.field(default_factory=set)


class PatternGenerator:
    def __init__(self, selection_type: int, selection_id: int, use_pos: bool = False):
        self.target_type = selection_type
        self.target_id = selection_id
        self.use_pos = use_pos
        self.essential_fields: Set[str] = set()
        self.pattern_data: Dict[str, Dict[str, Any]] = {}
        self.match_samples: List[Dict[str, Any]] = []
        self.varying_props: List[str] = []

        # 動的なフィールド調整
        self.check_fields = CHECK_FIELDS.copy()
        if self.use_pos:
            self.check_fields.append("pos_id")

    def _get_val_raw(self, cmd, field_name: str) -> Any:
        try:
            val = getattr(cmd, field_name)
            return val.value if isinstance(val, enum.Enum) else val
        except AttributeError:
            return None

    def _get_val_readable(self, cmd, field_name: str) -> str:
        try:
            val = getattr(cmd, field_name)
            if field_name == "card_id":
                if val == -1:
                    return "None"
                if val == 0:
                    return "Unknown"
                return CARD_MAP_REVERSE.get(val, f"CardID_{val}")
            return val.name if isinstance(val, enum.Enum) else str(val)
        except AttributeError:
            return "N/A"

    def _get_full_duel_card_info(self, state, table_index: int) -> str:
        duel_state = state.duel_state_data
        if table_index < 0 or table_index >= len(duel_state.duel_card_table):
            return ""
        card_obj = duel_state.duel_card_table[table_index]
        parts = []
        for field_info in dataclasses.fields(card_obj):
            val = getattr(card_obj, field_info.name)
            val_str = f"{val.value}({val.name})" if isinstance(val, enum.Enum) else str(val)
            parts.append(f"{field_info.name}:{val_str}")
        return " | ".join(parts)

    def _format_value_summary(self, values: List[Any]) -> str:
        unique_vals = list(set(values))
        if not unique_vals:
            return "N/A"
        if len(unique_vals) == 1:
            return str(unique_vals[0])
        if all(isinstance(v, (int, float)) for v in unique_vals):
            return f"{min(unique_vals)} ~ {max(unique_vals)}"
        if len(unique_vals) <= 3:
            return " or ".join(map(str, unique_vals))
        return f"{unique_vals[0]} ... {unique_vals[-1]} ({len(unique_vals)} variants)"

    def analyze(self):
        for pkl_file in config.DEMONSTRATION_DIR.glob("*.pkl"):
            try:
                with open(pkl_file, "rb") as f:
                    demos = pickle.load(f)
                    for d in demos:
                        req = d["state"].command_request
                        if int(req.selection_type) == self.target_type and int(req.selection_id) == self.target_id:
                            self.match_samples.append(
                                {"state": d["state"], "action": d["action"], "commands": req.commands}
                            )
            except Exception:
                continue

        if not self.match_samples:
            return False

        prop_values = {p: set() for p in SITUATION_PROPS}
        for sample in self.match_samples:
            sit = Situation(sample["state"])
            for p in SITUATION_PROPS:
                prop_values[p].add(getattr(sit, p))
        self.varying_props = [p for p in SITUATION_PROPS if len(prop_values[p]) > 1]

        for sample in self.match_samples:
            if len(sample["commands"]) > 1:
                for field in self.check_fields:
                    first_val = self._get_val_raw(sample["commands"][0], field)
                    if any(self._get_val_raw(c, field) != first_val for c in sample["commands"]):
                        self.essential_fields.add(field)
        self.essential_fields.add("card_id")
        if self.use_pos:
            self.essential_fields.add("pos_id")

        fields = sorted(list(self.essential_fields))
        for sample in self.match_samples:
            for cmd in sample["commands"]:
                identifier = "|".join([f"{f}:{self._get_val_readable(cmd, f)}" for f in fields])
                if identifier not in self.pattern_data:
                    self.pattern_data[identifier] = {f: self._get_val_raw(cmd, f) for f in fields}
        return True

    def _get_strategy_patterns(self) -> List[StrategyPattern]:
        fields = sorted(list(self.essential_fields))

        def get_ident(c):
            return "|".join([f"{f}:{self._get_val_readable(c, f)}" for f in fields])

        raw_groups: Dict[Tuple[str, Tuple[str, ...]], StrategyPattern] = {}
        for sample in self.match_samples:
            available = tuple(sorted([get_ident(c) for c in sample["commands"]]))
            chosen = get_ident(sample["action"].command_entry)
            key = (chosen, available)
            if key not in raw_groups:
                raw_groups[key] = StrategyPattern(available_idents=set(available), chosen_ident=chosen)
            raw_groups[key].samples.append(sample)

        patterns = sorted(list(raw_groups.values()), key=lambda x: len(x.available_idents))
        merged_patterns: List[StrategyPattern] = []
        for p in patterns:
            found_container = False
            for target in merged_patterns:
                if target.chosen_ident == p.chosen_ident:
                    if p.available_idents.issubset(target.available_idents):
                        target.samples.extend(p.samples)
                        target.is_merged = True
                        found_container = True
                        break
                    elif target.available_idents.issubset(p.available_idents):
                        diff = p.available_idents - target.available_idents
                        target.expanded_options.update(diff)
                        target.available_idents = p.available_idents
                        target.samples.extend(p.samples)
                        target.is_merged = True
                        found_container = True
                        break
            if not found_container:
                merged_patterns.append(p)
        return merged_patterns

    def save_as_python(self):
        handler_name = FILE_NAME_MAP.get(
            (self.target_type, self.target_id), f"selection_{self.target_type}_{self.target_id}"
        )
        output_dir = config.SRC_ROOT / "agents" / "noah" / "handlers" / "options"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{handler_name}.py"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f'"""Generated options for {handler_name}"""\n\nOPTIONS = {{\n')
            for k, v in sorted(self.pattern_data.items()):
                f.write(f'    "{k}": {v},\n')
            f.write("}\n")
        print(f"Generated OPTIONS: {file_path}")

    def save_as_llm_log(self):
        handler_name = FILE_NAME_MAP.get(
            (self.target_type, self.target_id), f"selection_{self.target_type}_{self.target_id}"
        )
        log_path = config.SRC_ROOT / "agents" / "noah" / "handlers" / "logs" / f"{handler_name}_human_logic.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        patterns = self._get_strategy_patterns()
        fields = sorted(list(self.essential_fields))

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                f"REVERSE ENGINEERING LOG (AGGREGATED): {handler_name} (Type:{self.target_type} ID:{self.target_id})\n"
                + "=" * 100
                + "\n"
            )
            f.write(f"Total Samples: {len(self.match_samples)} -> Strategy Patterns: {len(patterns)}\n")
            if self.use_pos:
                f.write("MODE: Position-Aware Hashing (pos_id included in identity)\n")

            for i, p in enumerate(patterns):
                f.write(f"\n### STRATEGY PATTERN {i+1} (Samples: {len(p.samples)}) ###\n")
                if p.is_merged:
                    f.write(f"NOTE: Aggregated samples with consistent decision despite choice subset variation.\n")
                    if p.expanded_options:
                        f.write(f"INFO: Options that didn't change the decision: {', '.join(p.expanded_options)}\n")

                f.write("[Aggregated Situation Properties]\n")
                for prop in self.varying_props:
                    vals = [getattr(Situation(s["state"]), prop) for s in p.samples]
                    # ID系プロパティの可読化
                    readable_vals = [
                        (
                            f"{CARD_MAP_REVERSE.get(v, v)}({v})"
                            if ("card_id" in prop or "monster_id" in prop) and isinstance(v, int) and v > 0
                            else v
                        )
                        for v in vals
                    ]
                    f.write(f"  [{prop}]: {self._format_value_summary(readable_vals)}\n")

                f.write("\n[Action Log History (Recent to Old)]\n")
                for back_index in range(1, 4):
                    history_summary = []
                    for s in p.samples:
                        sit = Situation(s["state"])
                        log = sit.get_log_entry(back_index)
                        if log and log.command:
                            name = CARD_MAP_REVERSE.get(log.command.card_id, f"ID:{log.command.card_id}")
                            action_name = FILE_NAME_MAP.get(
                                (log.selection_type, log.selection_id), f"T:{log.selection_type} I:{log.selection_id}"
                            )
                            history_summary.append(f"{name} ({action_name})")
                        else:
                            history_summary.append("None")
                    summary_str = self._format_value_summary(history_summary)
                    if summary_str != "None":
                        f.write(f"  - {back_index} action(s) ago: {summary_str}\n")

                f.write("\n[Aggregated Card-Specific Status]\n")
                card_methods = [
                    "has_card_in_hand",
                    "has_card_in_grave",
                    "has_card_on_mzone",
                    "has_card_on_szone",
                    "has_card_in_deck",
                    "get_chain_link_num",
                    "is_my_monster_equipped",
                    "is_rival_monster_equipped",
                    "is_enhanced_my_monster_exists",
                    "is_enhanced_rival_monster_exists",
                    "has_my_card_used_effect1",
                    "has_my_card_used_effect2",
                    "has_my_card_used_effect3",
                    "has_rival_card_used_effect1",
                    "has_rival_card_used_effect2",
                    "has_rival_card_used_effect3",
                    "has_rival_card_in_grave",
                    "has_rival_card_on_mzone",
                    "has_face_card_on_mzone",
                    "has_rival_face_card_on_mzone",
                    "has_rival_card_on_szone",
                    "has_card_in_chain_stack",
                    "was_action_taken_by_card",
                ]
                for card_name in CARD_MAP.keys():
                    res_list = []
                    for m in card_methods:
                        vals = [getattr(Situation(s["state"]), m)(card_name) for s in p.samples]
                        summary = self._format_value_summary(vals)
                        if summary not in ["False", "0", "-1"]:
                            res_list.append(f"{m}={summary}")
                    if res_list:
                        f.write(f"  {card_name} -> {', '.join(res_list)}\n")

                f.write(f"\n[COMMAND OPTIONS & HUMAN DECISION]\n")
                for ident in sorted(list(p.available_idents)):
                    mark = " ★SELECTED★ " if ident == p.chosen_ident else "            "
                    f.write(f"  {mark} {ident}\n")
                    ambiguous_details = set()
                    for s in p.samples:
                        matching_cmds = [
                            c
                            for c in s["commands"]
                            if "|".join([f"{f}:{self._get_val_readable(c, f)}" for f in fields]) == ident
                        ]
                        if len(matching_cmds) > 1:
                            for c in matching_cmds:
                                ambiguous_details.add(self._get_full_duel_card_info(s["state"], c.table_index))
                    if ambiguous_details:
                        f.write(
                            "                !! AMBIGUITY NOTICE: Multiple card instances share this identity. !!\n"
                        )
                        for detail in sorted(list(ambiguous_details)):
                            f.write(f"                >> FULL CARD DATA: {detail}\n")
                f.write("\n" + "-" * 100 + "\n")
        print(f"Generated GROUPED LOG for Gemini: {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=int, required=True)
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--pos", action="store_true", help="Include pos_id in command identity hash")
    args = parser.parse_args()
    gen = PatternGenerator(args.type, args.id, use_pos=args.pos)
    if gen.analyze():
        gen.save_as_python()
        gen.save_as_llm_log()


if __name__ == "__main__":
    main()
