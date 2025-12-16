import random
from typing import List, Optional

from ygo.models.command_request import CommandEntry
from ygo.models.duel_card import DuelCard
from ygo import constants as c

from src.env.state_data import StateData

# カードID定義 (atk.pyと共通のものも含む)
ID_CYCLONE = 1010
ID_HARPIE = 1007
ID_PREMATURE_BURIAL = 1009
ID_MONSTER_REBORN = 1025
ID_CALL_OF_HAUNTED = 1031
ID_RETURN_TO_FRONT = 1030
ID_POWERFUL_REBIRTH = 1016
ID_SILVERS_CRY = 1012
ID_MIRROR_FORCE = 1013
ID_TORRENTIAL_TRIBUTE = 1015
ID_DUST_TORNADO = 1014
ID_BOOK_OF_MOON = 1028
ID_LANCE = 1029
ID_SHRINK = 1011
ID_BLUE_EYES = 1004
ID_PALADIN = 1024
ID_GENESIS_DRAGON = 1023
ID_ASSAULT_WYVERN = 1005
ID_BOMBER_DRAGON = 1018
ID_MASKED_DRAGON = 1017
ID_KIDMODO = 1019


class OffExecutor:
    """
    相手ターンの行動ロジック (off.txt準拠)
    妨害・防御・エンドフェイズの除去を担当
    """

    def __init__(self) -> None:
        pass

    def select_action(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """
        相手ターンの行動を選択するメインメソッド
        """
        duel_state = state.duel_state_data
        general = duel_state.general_data
        request = state.command_request
        selection_type = request.selection_type

        # 1. 選択タイプによる分岐
        if selection_type == c.SelectionType.COMMAND:
            # 相手ターンのフェイズ進行に合わせた「発動確認」がここに来る
            # ACTIVATE, DECIDE (チェーン確認), CHANGE_PHASE (優先権放棄) など
            
            # チェーン確認(DECIDE)の場合は ChainExecutor に任せるのが基本だが、
            # agent.py の設計上、ここで DECIDE が来る可能性も考慮して YES/NO を返す
            if self._has_command_type(commands, c.CommandType.DECIDE):
                 # 簡易実装: 常にチェーンする方向で検討（詳細は chain.py だが、ここでも発動コマンドがあれば返す）
                 # 基本的には ACTIVATE コマンドがリストにあるはずなのでそちらを評価
                 pass

            if general.current_phase == c.Phase.DRAW:
                return self._phase_draw(commands)
            elif general.current_phase == c.Phase.STANDBY:
                return self._phase_standby(state, commands)
            elif general.current_phase == c.Phase.MAIN1:
                return self._phase_main1(state, commands)
            elif general.current_phase == c.Phase.BATTLE:
                return self._phase_battle(state, commands)
            elif general.current_phase == c.Phase.MAIN2:
                return self._phase_main2(state, commands)
            elif general.current_phase == c.Phase.END:
                return self._phase_end(state, commands)
            else:
                return self._pass_priority(commands)

        # 2. 対象選択・カード選択
        if selection_type in [c.SelectionType.SELECT_CARD, c.SelectionType.SELECT_UNSELECT_CARD]:
            return self._select_card_target(state, commands)

        # 3. 表示形式選択 (基本相手ターンには発生しないが、リビデ等で出る場合)
        if selection_type == c.SelectionType.SELECT_POSITION:
            return commands[0] # 守備表示優先などのロジックを入れる余地あり

        return None

    # =========================================================================
    # フェイズ別ロジック
    # =========================================================================

    def _phase_draw(self, commands: List[CommandEntry]) -> CommandEntry:
        # ドローフェイズは原則スルー
        return self._pass_priority(commands)

    def _phase_standby(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """
        相手スタンバイフェイズ
        off.txt : バック干渉、メイン前に価値が上がるS/Tを刈る
        """
        duel_state = state.duel_state_data
        
        # 砂塵の大竜巻・サイクロン
        # 優先順位: 装備/蘇生リンク > 永続 > 不明伏せ
        
        removal_cmd = self._logic_removal_activation(state, commands, phase="STANDBY")
        if removal_cmd:
            return removal_cmd

        # off.txt : 戦線復帰や銀龍の「壁が必要」な場合の先撃ち
        # 基本は相手メインの行動を見てからだが、LPが極端に低いならここで動く
        if duel_state.general_data.lp[c.PlayerId.MYSELF] <= 2000:
            revive_cmd = self._logic_emergency_defense(state, commands)
            if revive_cmd: return revive_cmd

        return self._pass_priority(commands)

    def _phase_main1(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """
        相手メインフェイズ1
        off.txt: 召喚反応（激流葬）、起動効果前の除去、召喚権使い切り待ち
        """
        duel_state = state.duel_state_data
        
        # 1. 激流葬 (Torrential Tribute) の判定
        # 相手がモンスターを召喚・特殊召喚したタイミングでの発動
        tt_cmd = self._find_command_by_card_id(commands, ID_TORRENTIAL_TRIBUTE, command_type=c.CommandType.ACTIVATE)
        if tt_cmd:
            opp_monsters = self._get_opponent_monsters(duel_state)
            opp_count = len(opp_monsters)
            
            # off.txt : 2体目/大型着地で総取り
            # マンジュ等のサーチ系単騎ならスルー
            # 青眼、パラディン等の大型なら1体でも撃つ
            
            is_boss_present = any(c.card_id in [ID_BLUE_EYES, ID_PALADIN, ID_GENESIS_DRAGON] for c in opp_monsters)
            
            if opp_count >= 2:
                return tt_cmd
            elif is_boss_present:
                return tt_cmd
            elif opp_count == 1:
                # 1体だけ。相手がさらに展開しそうなら温存。
                # 確率で「待ち」を入れる (80%待ち、20%で単発除去 - 事故狙い)
                if random.random() < 0.2:
                    return tt_cmd
            
            # 基本は温存してパス（相手の展開を待つ）
            
        # 2. 除去 (砂塵/サイク)
        # メイン開始直後や、永続発動に対して
        removal_cmd = self._logic_removal_activation(state, commands, phase="MAIN1")
        if removal_cmd:
            return removal_cmd

        return self._pass_priority(commands)

    def _phase_battle(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """
        バトルフェイズ
        off.txt: スタートステップの壁生成、攻撃宣言時のミラフォ、ダメステの収縮/月書
        """
        duel_state = state.duel_state_data
        
        # 1. スタートステップ / 攻撃宣言前の壁生成
        # ライフが危険で壁がないなら、バトルフェイズに入った瞬間に蘇生
        if duel_state.general_data.lp[c.PlayerId.MYSELF] <= 3000 and len(self._get_my_monsters(duel_state)) == 0:
             revive_cmd = self._logic_emergency_defense(state, commands)
             if revive_cmd: return revive_cmd

        # 2. ミラーフォース (攻撃宣言時)
        mf_cmd = self._find_command_by_card_id(commands, ID_MIRROR_FORCE, command_type=c.CommandType.ACTIVATE)
        if mf_cmd:
            opp_atk_monsters = [m for m in self._get_opponent_monsters(duel_state) if m.position == c.PosId.MZONE and m.face == c.Face.FACEUP_ATTACK]
            
            # off.txt : 複数残っている or 致命打(2500以上) or LP危険
            if len(opp_atk_monsters) >= 2:
                return mf_cmd
            
            has_high_atk = any(m.atk_val >= 2500 for m in opp_atk_monsters)
            is_lethal = sum(m.atk_val for m in opp_atk_monsters) >= duel_state.general_data.lp[c.PlayerId.MYSELF]
            
            if has_high_atk or is_lethal:
                return mf_cmd
                
            # 1体だけの低打点なら温存 (スルー)
        
        # 3. 月の書 / 収縮 / 聖槍 (バトルステップ/ダメージステップ)
        # 攻撃を止める、または返り討ちにする
        
        # 月の書
        book_cmd = self._find_command_by_card_id(commands, ID_BOOK_OF_MOON, command_type=c.CommandType.ACTIVATE)
        if book_cmd:
            # 相手の攻撃モンスターが厄介なら裏にする (攻撃無効化)
            # 攻撃モンスターの特定は難しいが、最も攻撃力が高いモンスターが殴っていると仮定、あるいは対象選択時に判断
            # ここでは「発動できるなら、相手に高打点がいれば使う」ロジック
            opp_monsters = self._get_opponent_monsters(duel_state)
            dangerous = [m for m in opp_monsters if m.atk_val >= 1900]
            if dangerous:
                return book_cmd

        # 収縮 (ダメステ発動可)
        shrink_cmd = self._find_command_by_card_id(commands, ID_SHRINK, command_type=c.CommandType.ACTIVATE)
        if shrink_cmd:
            # 自分が戦闘破壊されそうで、収縮を使えば勝てる場合
            # (詳細な戦闘シミュレーションは困難なので、ヒューリスティックに「自分のモンスターがいるなら使う」)
            if len(self._get_my_monsters(duel_state)) > 0:
                return shrink_cmd

        return self._pass_priority(commands)

    def _phase_main2(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """
        相手メインフェイズ2
        off.txt: ほぼメイン1と同じだが、エンドフェイズ割りの計画を立てる
        """
        # 基本的にスルーしてエンドフェイズに回すが、
        # 相手が厄介な永続を貼ってきたら即割る
        removal_cmd = self._logic_removal_activation(state, commands, phase="MAIN2")
        if removal_cmd:
            return removal_cmd
            
        return self._pass_priority(commands)

    def _phase_end(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """
        相手エンドフェイズ
        off.txt: エンドサイクの主戦場
        """
        duel_state = state.duel_state_data
        
        # 1. サイクロン / 砂塵 (エンド割り)
        # off.txt : 不明伏せはエンドで叩く
        removal_cmd = self._logic_removal_activation(state, commands, phase="END")
        if removal_cmd:
            return removal_cmd
            
        # 2. フリーチェーン蘇生 (次ターンの攻め手確保)
        # off.txt : 次ターンリーサル圏内ならエンドに蘇生
        # ここでは常に「出せるなら出す」ではなく、リソース温存も考慮
        # 墓地に青眼などがいれば蘇生しておく
        revive_ids = [ID_SILVERS_CRY, ID_CALL_OF_HAUNTED, ID_RETURN_TO_FRONT, ID_POWERFUL_REBIRTH]
        for rid in revive_ids:
            cmd = self._find_command_by_card_id(commands, rid, command_type=c.CommandType.ACTIVATE)
            if cmd:
                # 自分の場が空、または次ターン攻めたい
                # 確率 70% で発動 (相手の除去を警戒して少し控える等のゆらぎ)
                if len(self._get_my_monsters(duel_state)) <= 2 and random.random() < 0.7:
                    return cmd

        return self._pass_priority(commands)

    # =========================================================================
    # ロジック詳細
    # =========================================================================

    def _logic_removal_activation(self, state: StateData, commands: List[CommandEntry], phase: str) -> Optional[CommandEntry]:
        """
        サイクロン・砂塵の発動判断 (Phase共通)
        """
        duel_state = state.duel_state_data
        opp_field = [c for c in duel_state.duel_card_table if c.player_id == c.PlayerId.RIVAL]
        opp_backrow = [c for c in opp_field if c.position == c.PosId.SZONE]
        
        # 発動候補
        cyclone = self._find_command_by_card_id(commands, ID_CYCLONE, command_type=c.CommandType.ACTIVATE)
        dust = self._find_command_by_card_id(commands, ID_DUST_TORNADO, command_type=c.CommandType.ACTIVATE)
        
        if not (cyclone or dust):
            return None
            
        # ターゲット評価
        # 表側の永続系 (装備、永続罠、フィールド) を最優先
        faceup_spells = [c for c in opp_backrow if c.face == c.Face.FACEUP]
        set_cards = [c for c in opp_backrow if c.face == c.Face.DOWN]
        
        has_faceup_target = len(faceup_spells) > 0
        has_set_target = len(set_cards) > 0
        
        # スタンバイ/メイン: 表側があるなら即割る
        if phase in ["STANDBY", "MAIN1", "MAIN2"]:
            if has_faceup_target:
                return dust if dust else cyclone
            # スタンバイなら伏せも割って展開を阻害しにいく (確率)
            if phase == "STANDBY" and has_set_target and random.random() < 0.5:
                return dust if dust else cyclone

        # エンドフェイズ: 伏せ除去の本番
        if phase == "END":
            # 伏せがあるなら積極的に割る
            if has_set_target:
                return dust if dust else cyclone
            # 表側ももちろん割る
            if has_faceup_target:
                return dust if dust else cyclone
                
        return None

    def _logic_emergency_defense(self, state: StateData, commands: List[CommandEntry]) -> Optional[CommandEntry]:
        """
        緊急時の壁生成 (戦線復帰、リビデ、銀龍)
        """
        revive_ids = [ID_RETURN_TO_FRONT, ID_SILVERS_CRY, ID_CALL_OF_HAUNTED]
        for rid in revive_ids:
            cmd = self._find_command_by_card_id(commands, rid, command_type=c.CommandType.ACTIVATE)
            if cmd:
                return cmd
        return None

    def _select_card_target(self, state: StateData, commands: List[CommandEntry]) -> CommandEntry:
        """
        対象選択ロジック (除去対象、蘇生対象など)
        """
        duel_state = state.duel_state_data
        
        # 文脈がわからないため、コマンドリストの内容から推測
        # 相手フィールドのカードが含まれる -> 除去ターゲット選択
        # 自分墓地のカードが含まれる -> 蘇生ターゲット選択
        
        # 1. 除去ターゲット (相手の表側永続 > 相手の伏せ)
        opp_faceup = []
        opp_set = []
        
        for cmd in commands:
            # table_index からカード情報を取得して判定
            # (簡易実装: IDリストマッチングなどは省略し、表示形式だけで判断)
            # ※本来は duel_state.duel_card_table[cmd.table_index] を見る
            pass

        # 簡易ロジック: リストの先頭を選ぶのではなく、ランダム性を持たせて「読み」を演出
        # ただし、明らかに強いカード（青眼など）が対象候補にあればそれを選ぶ
        
        priority_targets = [ID_BLUE_EYES, ID_PALADIN, ID_PREMATURE_BURIAL, ID_CALL_OF_HAUNTED]
        for pid in priority_targets:
            cmd = self._find_command_by_card_id(commands, pid)
            if cmd: return cmd
            
        return commands[0]

    # =========================================================================
    # ヘルパーメソッド
    # =========================================================================

    def _get_my_monsters(self, duel_state: StateData) -> List[DuelCard]:
        return [c for c in duel_state.duel_card_table 
                if c.player_id == c.PlayerId.MYSELF and c.position == c.PosId.MZONE]

    def _get_opponent_monsters(self, duel_state: StateData) -> List[DuelCard]:
        return [c for c in duel_state.duel_card_table 
                if c.player_id == c.PlayerId.RIVAL and c.position == c.PosId.MZONE]

    def _find_command_by_card_id(self, commands: List[CommandEntry], card_id: int, command_type: int = None) -> Optional[CommandEntry]:
        for cmd in commands:
            if command_type is not None and cmd.command_type != command_type:
                continue
            if cmd.card_id == card_id:
                return cmd
        return None

    def _has_command_type(self, commands: List[CommandEntry], command_type: int) -> bool:
        return any(cmd.command_type == command_type for cmd in commands)

    def _pass_priority(self, commands: List[CommandEntry]) -> CommandEntry:
        """優先権放棄（パスまたはフェイズ移行）を探して返す"""
        # PASS
        for cmd in commands:
            if cmd.command_type == c.CommandType.PASS:
                return cmd
        # CHANGE_PHASE (相手ターン中は実質パス)
        for cmd in commands:
            if cmd.command_type == c.CommandType.CHANGE_PHASE:
                return cmd
        # No / Cancel (DECIDE)
        for cmd in commands:
            if cmd.command_type == c.CommandType.DECIDE:
                # 一般的に index 1 が No/Cancel
                if commands.index(cmd) == 1: 
                    return cmd
        
        return commands[0]